"""The one-step setup flow: backend choice, run configuration, engine
resolution, the whole `coli setup` against a fake hub, and start/status/stop.

No network, no real model, no real build: the hub is tests/fake_hub.py, the
hardware report is canned, the engine is a file that names the Vulkan loader,
and the server is a twenty-line stand-in that answers /health like `coli web`
does and writes the same pidfile, so the real `coli stop` can stop it.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
C_DIR = TESTS.parent
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(TESTS))
import setup_catalog  # noqa: E402
import setup_download  # noqa: E402
import setup_flow  # noqa: E402
import setup_hw  # noqa: E402
from fake_hub import FakeHub  # noqa: E402
from family_registry import family_by_id  # noqa: E402

REPO = "tester/tiny-model"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def hw_report(vulkan=None, nvidia=(), icd=None, os_id="ubuntu"):
    return {
        "os": {"platform": "linux", "machine": "x86_64", "wsl": False, "id": os_id,
               "id_like": "debian", "pretty_name": "Ubuntu 24.04"},
        "cpu": {"name": "Test CPU", "features": ["avx2"], "physical_cores": 6, "logical_cores": 12,
                "arch": "x86_64"},
        "memory": {"total": 27 * 10**9, "available": 24 * 10**9},
        "disk": None, "vulkan": {"devices": [vulkan] if vulkan else [], "icd": icd},
        "nvidia": list(nvidia), "windows_video": [],
        "gpu": {"vulkan": vulkan, "vulkan_icd": icd, "nvidia": list(nvidia),
                "has_gpu": bool(vulkan or nvidia)},
    }


IGPU = {"name": "Iris Xe", "type": "integrated", "api_version": "1.2.318",
        "api_version_raw": (1 << 22) | (2 << 12) | 318}
RTX = {"index": 0, "name": "NVIDIA GeForce RTX 4070", "total_bytes": 12 * 2**30,
       "free_bytes": 11 * 2**30, "driver": "560"}
TC_ALL = {"source_checkout": True, "make": "/usr/bin/make", "cc": "/usr/bin/gcc",
          "glslc": "/usr/bin/glslc", "vulkan_headers": True, "nvcc": "/usr/local/cuda/bin/nvcc",
          "msys2": None, "npm": None, "can_build": True, "can_build_vulkan": True,
          "can_build_cuda": True}


def setup_args(**overrides):
    values = dict(yes=True, json=False, pick=None, model_dir=None, dir=None, backend=None,
                  no_gpu=False, host=None, port=None, no_start=True, background=False,
                  no_browser=True, reconfigure=False, list=False, all=False, via_windows="no",
                  no_verify=False)
    values.update(overrides)
    return argparse.Namespace(**values)


class HomeTestCase(unittest.TestCase):
    """Each test gets its own setup folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = os.path.join(self.tmp.name, "home")
        patcher = mock.patch.dict(os.environ, {"COLI_SETUP_HOME": self.home})
        patcher.start()
        self.addCleanup(patcher.stop)


class BackendChoice(unittest.TestCase):
    def choose(self, hw, family="qwen36", tc=None, requested="auto", platform="linux"):
        with mock.patch.object(setup_flow.sys, "platform", platform):
            return setup_flow.choose_backend(hw, family_by_id(family), tc or TC_ALL, requested)

    def test_no_gpu_is_cpu(self):
        decision = self.choose(hw_report())
        self.assertEqual(decision["backend"], "cpu")
        self.assertIn("no GPU", decision["reason"])

    def test_integrated_vulkan_gpu(self):
        decision = self.choose(hw_report(vulkan=IGPU))
        self.assertEqual(decision["backend"], "vulkan")
        self.assertEqual(decision["gpu"], "Iris Xe")

    def test_nvidia_with_toolkit_takes_cuda(self):
        self.assertEqual(self.choose(hw_report(vulkan=IGPU, nvidia=[RTX]))["backend"], "cuda")

    def test_nvidia_without_toolkit_falls_to_vulkan_and_says_how(self):
        tc = dict(TC_ALL, nvcc=None, can_build_cuda=False)
        decision = self.choose(hw_report(vulkan=IGPU, nvidia=[RTX]), tc=tc)
        self.assertEqual(decision["backend"], "vulkan")
        self.assertEqual(decision["missing"][0][0], "cuda")
        self.assertIn("nvidia-cuda-toolkit", decision["missing"][0][1])

    def test_engine_without_cuda_path_uses_vulkan(self):
        decision = self.choose(hw_report(vulkan=IGPU, nvidia=[RTX]), family="mimo")
        self.assertEqual(decision["backend"], "vulkan")

    def test_vulkan_gpu_without_headers_prints_the_package_command(self):
        tc = dict(TC_ALL, vulkan_headers=False, glslc=None, can_build_vulkan=False)
        decision = self.choose(hw_report(vulkan=IGPU), tc=tc)
        self.assertEqual(decision["backend"], "cpu")
        self.assertEqual(decision["missing"],
                         [("vulkan", "sudo apt install libvulkan-dev glslc mesa-vulkan-drivers")])

    def test_cpu_when_asked(self):
        self.assertEqual(self.choose(hw_report(vulkan=IGPU, nvidia=[RTX]), requested="cpu")["backend"],
                         "cpu")

    def test_windows_without_msys2_points_at_it(self):
        tc = dict(TC_ALL, source_checkout=False, make=None, cc=None, can_build=False,
                  can_build_vulkan=False, can_build_cuda=False)
        decision = self.choose(hw_report(vulkan=IGPU), tc=tc, platform="win32")
        self.assertEqual(decision["backend"], "cpu")
        hint = decision["missing"][0][1]
        self.assertIn("msys2.org", hint)
        self.assertIn("mingw-w64-ucrt-x86_64-gcc", hint)
        self.assertIn("mingw-w64-ucrt-x86_64-shaderc", hint)
        self.assertIn("source checkout", hint)      # a release archive cannot rebuild itself


class PackageHints(unittest.TestCase):
    def test_one_command_per_system(self):
        hint = setup_flow.package_hint
        self.assertEqual(hint(["build", "vulkan"], {"ID": "ubuntu", "ID_LIKE": "debian"}, "linux"),
                         "sudo apt install build-essential libvulkan-dev glslc mesa-vulkan-drivers")
        self.assertEqual(hint("vulkan", {"id": "fedora"}, "linux"),
                         "sudo dnf install vulkan-loader-devel glslc mesa-vulkan-drivers")
        self.assertIn("sudo pacman -S --needed vulkan-icd-loader", hint("vulkan", {"id": "arch"}, "linux"))
        self.assertIn("Vulkan driver", hint("vulkan", {"id": "arch"}, "linux"))
        self.assertIn("developer.nvidia.com", hint("cuda", {"id": "fedora"}, "linux"))
        self.assertIn("xcode-select", hint("build", {}, "darwin"))
        self.assertIn("from your distribution", hint("vulkan", {"id": "gentoo"}, "linux"))


class RunConfiguration(HomeTestCase):
    def test_vulkan_environment(self):
        engine_dir = os.path.join(self.tmp.name, "engines")
        os.makedirs(os.path.join(engine_dir, "shaders"))
        hw = hw_report(vulkan=IGPU, icd="/home/u/dzn_icd.json")
        env = setup_flow.run_environment("vulkan", os.path.join(engine_dir, "qwen36"),
                                         family_by_id("qwen36"), hw)
        self.assertEqual(env["COLI_VULKAN"], "1")
        self.assertEqual(env["COLI_VK_SHADERS"], os.path.join(engine_dir, "shaders"))
        self.assertEqual(env["VK_ICD_FILENAMES"], "/home/u/dzn_icd.json")
        self.assertNotIn("COLI_NO_OMP_TUNE", env)
        glm = setup_flow.run_environment("vulkan", os.path.join(engine_dir, "colibri"),
                                         family_by_id("glm"), hw)
        self.assertEqual(glm["COLI_NO_OMP_TUNE"], "1")
        self.assertEqual(glm["OMP_NUM_THREADS"], "6")     # physical cores, not 12 threads
        self.assertEqual(setup_flow.run_environment("cpu", "x", family_by_id("qwen36"), hw), {})

    def test_cuda_arguments(self):
        args = setup_flow.launcher_args("cuda", "/m", "127.0.0.1", 8000)
        self.assertEqual(args[:2], ["web", "--model"])
        self.assertIn("--auto-tier", args)
        self.assertEqual(args[args.index("--gpu") + 1], "auto")
        self.assertNotIn("--gpu", setup_flow.launcher_args("vulkan", "/m", "127.0.0.1", 8000))

    def test_written_and_read_back(self):
        entry = setup_catalog.by_id("qwen36-35b")
        engine_info = {"launcher_dir": str(C_DIR), "engine": str(C_DIR / "qwen36"),
                       "backend": "vulkan", "source": "built"}
        cfg = setup_flow.make_config(model_dir="/models/qwen36-35b", family=family_by_id("qwen36"),
                                     entry=entry, engine_info=engine_info,
                                     hw=hw_report(vulkan=IGPU), host="0.0.0.0", port=8123)
        setup_flow.save_config(cfg)
        loaded = setup_flow.load_config()
        self.assertEqual(loaded["model"]["repo"], entry.repo)
        self.assertEqual(loaded["launcher"], str(C_DIR / "coli"))
        self.assertEqual(loaded["env"]["COLI_VULKAN"], "1")
        self.assertEqual(loaded["urls"], {"browser": "http://127.0.0.1:8123/",
                                          "openai_base_url": "http://127.0.0.1:8123/v1",
                                          "anthropic_base_url": "http://127.0.0.1:8123"})
        self.assertIn("COLI_VULKAN=1", setup_flow.equivalent_command(loaded))
        self.assertIn("--port 8123", setup_flow.equivalent_command(loaded))
        self.assertEqual(setup_flow.configured_port(), 8123)
        # an unknown layout is not trusted
        Path(setup_flow.config_path()).write_text(json.dumps({"version": 99}))
        self.assertIsNone(setup_flow.load_config())
        self.assertEqual(setup_flow.configured_port(), 8000)

    def test_last_turn_speed(self):
        self.assertEqual(setup_flow.last_turn_speed({"turns": [{"wall_s": 4.0, "completion_tokens": 10}]}),
                         2.5)
        self.assertIsNone(setup_flow.last_turn_speed({"turns": []}))


class EngineResolution(HomeTestCase):
    def test_binary_backend(self):
        for content, expected in ((b"\0libvulkan.so.1\0", "vulkan"), (b"VULKAN-1.DLL", "vulkan"),
                                  (b"libcudart.so.12", "cuda"), (b"plain", "cpu")):
            path = Path(self.tmp.name, "engine")
            path.write_bytes(b"\x7fELF" + content)
            self.assertEqual(setup_flow.binary_backend(str(path)), expected)

    def test_present_engine_with_the_right_backend_is_not_rebuilt(self):
        Path(self.tmp.name, "qwen36").write_bytes(b"libvulkan.so.1")
        decision = {"backend": "vulkan", "missing": []}
        with mock.patch.object(setup_flow, "HERE", self.tmp.name), \
             mock.patch.object(setup_flow, "build_engine") as build:
            info = setup_flow.resolve_engine(family_by_id("qwen36"), None, decision, TC_ALL,
                                             out=lambda *_: None)
        build.assert_not_called()
        self.assertEqual(info["source"], "present")

    def test_a_cpu_engine_is_rebuilt_for_vulkan(self):
        Path(self.tmp.name, "qwen36").write_bytes(b"plain")
        decision = {"backend": "vulkan", "missing": []}
        with mock.patch.object(setup_flow, "HERE", self.tmp.name), \
             mock.patch.object(setup_flow, "build_engine", return_value="/built/qwen36") as build:
            info = setup_flow.resolve_engine(family_by_id("qwen36"), None, decision, TC_ALL,
                                             out=lambda *_: None)
        build.assert_called_once()
        self.assertEqual(info["source"], "built")

    def test_make_command(self):
        with mock.patch.object(setup_flow.sys, "platform", "linux"), \
             mock.patch.dict(os.environ, {"ARCH": ""}):
            cmd, _cwd, _env = setup_flow.make_command(family_by_id("deepseek_v4"), "vulkan", TC_ALL)
            self.assertEqual(cmd[-3:], ["deepseek-v4", "ARCH=native", "VK=1"])
            cmd, _cwd, _env = setup_flow.make_command(family_by_id("qwen36"), "cuda", TC_ALL)
            self.assertIn("CUDA=1", cmd)
            self.assertIn("CUDA_ARCH=native", cmd)


class Releases(HomeTestCase):
    def test_asset_names(self):
        suffix = setup_flow.release_asset_suffix
        self.assertEqual(suffix("linux", "x86_64"), "linux-x86_64.tar.gz")
        self.assertEqual(suffix("win32", "AMD64"), "windows-x86_64.zip")
        self.assertEqual(suffix("darwin", "arm64"), "macos-arm64.tar.gz")
        self.assertIsNone(suffix("linux", "aarch64"))

    def test_find_release_falls_back_to_latest(self):
        seen = []

        def getter(url):
            seen.append(url)
            if url.endswith("/tags/v9.9.9"):
                raise setup_flow.urllib.error.URLError("404")
            return {"tag_name": "v1.12.1", "assets": [{"name": "a.zip", "browser_download_url": "u"}]}

        release = setup_flow.find_release("9.9.9", api="https://api.example/releases", getter=getter)
        self.assertEqual(release, {"tag": "v1.12.1", "assets": {"a.zip": "u"}})
        self.assertEqual(seen, ["https://api.example/releases/tags/v9.9.9",
                                "https://api.example/releases/latest"])

    def test_sha256sums(self):
        sums = setup_flow.parse_sha256sums("a" * 64 + "  colibri-v1-linux.tar.gz\n" +
                                           "B" * 64 + " *other.zip\nnot a line\n")
        self.assertEqual(sums, {"colibri-v1-linux.tar.gz": "a" * 64, "other.zip": "b" * 64})

    def test_archives_cannot_escape(self):
        evil_zip = os.path.join(self.tmp.name, "evil.zip")
        with zipfile.ZipFile(evil_zip, "w") as zf:
            zf.writestr("../escape.txt", "x")
        with self.assertRaises(setup_flow.SetupError):
            setup_flow.extract_archive(evil_zip, os.path.join(self.tmp.name, "out"))
        evil_tar = os.path.join(self.tmp.name, "evil.tar.gz")
        with tarfile.open(evil_tar, "w:gz") as tf:
            info = tarfile.TarInfo("/etc/escape")
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
        with self.assertRaises(setup_flow.SetupError):
            setup_flow.extract_archive(evil_tar, os.path.join(self.tmp.name, "out2"))
        self.assertFalse(Path(self.tmp.name, "escape.txt").exists())

    def serve_release(self, files):
        """A local release: the archive and SHA256SUMS.txt over HTTP."""
        folder = os.path.join(self.tmp.name, "release")
        os.makedirs(folder)
        for name, data in files.items():
            Path(folder, name).write_bytes(data)
        class Quiet(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=folder))
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                         daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        return {name: f"{base}/{name}" for name in files}

    def make_archive(self, tag):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
            for name, data in (("coli", b"# launcher"), ("qwen36", b"engine"),
                               ("web/dist/index.html", b"<html></html>"), ("tools/k3_tokenizer.py", b"")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return buffer.getvalue()

    def test_prebuilt_engine_when_there_is_no_compiler(self):
        tag = "v1.12.1"
        name = f"colibri-{tag}-linux-x86_64.tar.gz"
        archive = self.make_archive(tag)
        sums = f"{hashlib.sha256(archive).hexdigest()}  {name}\n".encode()
        urls = self.serve_release({name: archive, "SHA256SUMS.txt": sums})
        getter = lambda url: {"tag_name": tag, "assets": [{"name": n, "browser_download_url": u}
                                                          for n, u in urls.items()]}
        tc = dict(TC_ALL, source_checkout=False, can_build=False, can_build_vulkan=False)
        empty = os.path.join(self.tmp.name, "nothing-here")
        os.makedirs(empty)
        with mock.patch.object(setup_flow, "HERE", empty), \
             mock.patch.object(setup_flow, "release_asset_suffix", return_value="linux-x86_64.tar.gz"), \
             mock.patch.object(setup_flow, "_api_get", side_effect=getter):
            out = []
            info = setup_flow.resolve_engine(family_by_id("qwen36"), setup_catalog.by_id("qwen36-35b"),
                                             {"backend": "vulkan", "missing": []}, tc, out=out.append)
            self.assertEqual(info["backend"], "cpu")               # prebuilt engines are CPU builds
            self.assertTrue(info["source"].startswith("release v1.12.1"))
            self.assertTrue(os.path.isfile(info["engine"]))
            self.assertTrue(os.path.isfile(os.path.join(info["launcher_dir"], "web", "dist", "index.html")))
            # A model newer than the release is refused with the way forward.
            with self.assertRaises(setup_flow.SetupError) as caught:
                setup_flow.resolve_engine(family_by_id("qwen36"), setup_catalog.by_id("qwen3-coder-30b"),
                                          {"backend": "cpu", "missing": []}, tc, out=out.append)
            self.assertIn("predates", str(caught.exception))

    def test_bad_checksum_is_refused(self):
        tag = "v1.12.1"
        name = f"colibri-{tag}-linux-x86_64.tar.gz"
        urls = self.serve_release({name: self.make_archive(tag),
                                   "SHA256SUMS.txt": f"{'0' * 64}  {name}\n".encode()})
        getter = lambda url: {"tag_name": tag, "assets": [{"name": n, "browser_download_url": u}
                                                          for n, u in urls.items()]}
        with mock.patch.object(setup_flow, "release_asset_suffix", return_value="linux-x86_64.tar.gz"):
            with self.assertRaises(setup_flow.SetupError) as caught:
                setup_flow.fetch_release_archive("1.12.1", out=lambda *_: None, getter=getter)
        self.assertIn("checksum", str(caught.exception))


class WholeSetup(HomeTestCase):
    """`coli setup --yes` from nothing to a written configuration, then a rerun."""

    def setUp(self):
        super().setUp()
        self.src = os.path.join(self.tmp.name, "repo")
        os.makedirs(self.src)
        Path(self.src, "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe_text"}))
        Path(self.src, "tokenizer.json").write_text("{}")
        self.shard = os.urandom(400_000)
        Path(self.src, "model-00000.safetensors").write_bytes(self.shard)
        self.hub = FakeHub({REPO: self.src})
        self.hub.__enter__()
        self.addCleanup(self.hub.__exit__)
        catalog = os.path.join(self.tmp.name, "catalog.json")
        Path(catalog).write_text(json.dumps([{
            "id": "tiny", "family": "qwen36", "name": "Tiny test model", "repo": REPO,
            "disk_gb": 0.001, "ram_min_gb": 0.1, "ram_good_gb": 0.2, "dense_gb": 0.01,
            "rank": 99, "size_class": "small", "summary": "fixture"}]))
        self.engines = os.path.join(self.tmp.name, "c")
        os.makedirs(os.path.join(self.engines, "shaders"))
        os.makedirs(os.path.join(self.engines, "web", "dist"))
        Path(self.engines, "web", "dist", "index.html").write_text("<html></html>")
        Path(self.engines, "qwen36").write_bytes(b"\x7fELF libvulkan.so.1")
        Path(self.engines, "coli").write_text("# launcher\n")
        self.models = os.path.join(self.tmp.name, "models")
        for patcher in (
                mock.patch.dict(os.environ, {"HF_ENDPOINT": self.hub.base,
                                             "COLI_SETUP_CATALOG": catalog}),
                mock.patch.object(setup_flow, "HERE", self.engines),
                mock.patch.object(setup_hw, "detect",
                                  return_value=hw_report(vulkan=IGPU, icd="/home/u/dzn.json")),
                mock.patch.object(setup_hw, "is_wsl", return_value=False),
                mock.patch.object(setup_flow, "toolchain", return_value=TC_ALL),
                mock.patch.object(setup_flow, "plan_summary",
                                  return_value=("dense part 0.0 GB in RAM", [])),
                mock.patch.object(setup_download.time, "sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_setup(self, **overrides):
        out = io.StringIO()
        ui = setup_flow.UI(False, stream=out)
        code = setup_flow.run(lambda a: setup_flow.cmd_setup(a, ui=ui),
                              setup_args(dir=self.models, **overrides))
        return code, out.getvalue()

    def test_install_then_rerun(self):
        code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        model_dir = os.path.join(self.models, "tiny")
        self.assertEqual(Path(model_dir, "model-00000.safetensors").read_bytes(), self.shard)
        self.assertTrue(setup_download.is_complete(model_dir))
        cfg = setup_flow.load_config()
        self.assertEqual(cfg["status"], "ready")
        self.assertEqual(cfg["backend"], "vulkan")
        self.assertEqual(cfg["env"]["COLI_VULKAN"], "1")
        self.assertEqual(cfg["env"]["VK_ICD_FILENAMES"], "/home/u/dzn.json")
        self.assertEqual(cfg["args"][:3], ["web", "--model", model_dir])
        self.assertEqual(cfg["launcher"], os.path.join(self.engines, "coli"))
        self.assertIn("Saved the run configuration", text)
        self.assertIn("OpenAI base URL:     http://127.0.0.1:8000/v1", text)
        self.assertIn("Anthropic base URL:  http://127.0.0.1:8000", text)
        # Rerun: straight to the start, nothing fetched, nothing built.
        before = len(self.hub.requests)
        with mock.patch.object(setup_flow, "build_engine") as build:
            code, text = self.run_setup()
        self.assertEqual(code, 0, text)
        self.assertIn("Already set up: Tiny test model", text)
        self.assertEqual(len(self.hub.requests), before)
        build.assert_not_called()

    def test_interrupted_download_continues_on_rerun(self):
        real = setup_download.download_repo
        self.hub.cut_after["model-00000.safetensors"] = 150_000
        with mock.patch.object(setup_download, "download_repo",
                               side_effect=lambda *a, **k: real(*a, **dict(k, retries=0))):
            code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 1)
        self.assertEqual(setup_flow.load_config()["status"], "pending")
        self.assertEqual(setup_flow.read_state()["phase"], "error")
        part = Path(self.models, "tiny", "model-00000.safetensors.part")
        self.assertEqual(part.stat().st_size, 150_000)
        # Same command again, without naming the model: it resumes the pending one.
        code, text = self.run_setup()
        self.assertEqual(code, 0, text)
        self.assertIn("Resuming the setup of Tiny test model", text)
        self.assertIn("resuming:", text)
        ranges = [r["headers"].get("Range") for r in self.hub.requests
                  if r["path"].endswith("model-00000.safetensors") and "/cdn/" in r["path"]]
        self.assertEqual(ranges[-1], "bytes=150000-")
        self.assertEqual(Path(self.models, "tiny", "model-00000.safetensors").read_bytes(), self.shard)
        self.assertEqual(setup_flow.load_config()["status"], "ready")

    def test_a_missing_engine_is_rebuilt_without_asking_again(self):
        code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        os.remove(os.path.join(self.engines, "qwen36"))
        shard_fetches = sum("/cdn/" in r["path"] for r in self.hub.requests)

        def rebuild(family, backend, tc, out=print):
            Path(self.engines, "qwen36").write_bytes(b"libvulkan.so.1")
            return os.path.join(self.engines, "qwen36")

        with mock.patch.object(setup_flow, "build_engine", side_effect=rebuild) as build:
            code, text = self.run_setup()
        self.assertEqual(code, 0, text)
        self.assertIn("Resuming the setup of Tiny test model", text)
        self.assertIn("Download: already complete", text)
        build.assert_called_once()
        self.assertEqual(sum("/cdn/" in r["path"] for r in self.hub.requests), shard_fetches)

    def test_a_rerun_switches_to_the_gpu_once_its_packages_are_there(self):
        no_vulkan = dict(TC_ALL, vulkan_headers=False, glslc=None, can_build_vulkan=False)
        Path(self.engines, "qwen36").write_bytes(b"\x7fELF plain")          # a CPU build
        with mock.patch.object(setup_flow, "toolchain", return_value=no_vulkan):
            code, text = self.run_setup(pick="tiny")
        self.assertEqual(code, 0, text)
        self.assertIn("to use the GPU through VULKAN, first run:", text)
        cfg = setup_flow.load_config()
        self.assertEqual((cfg["backend"], cfg["gpu_pending"]), ("cpu", ["vulkan"]))
        # Still no packages: a rerun only starts.
        with mock.patch.object(setup_flow, "toolchain", return_value=no_vulkan):
            code, text = self.run_setup()
        self.assertIn("Already set up", text)

        def rebuild(family, backend, tc, out=print):
            self.assertEqual(backend, "vulkan")
            Path(self.engines, "qwen36").write_bytes(b"libvulkan.so.1")
            return os.path.join(self.engines, "qwen36")

        with mock.patch.object(setup_flow, "build_engine", side_effect=rebuild):
            code, text = self.run_setup()                  # the packages are here now
        self.assertEqual(code, 0, text)
        self.assertIn("rebuilding the engine for the GPU", text)
        cfg = setup_flow.load_config()
        self.assertEqual((cfg["backend"], cfg["gpu_pending"]), ("vulkan", []))
        self.assertEqual(cfg["env"]["COLI_VULKAN"], "1")

    def test_wsl_download_through_windows_when_faster(self):
        entry = setup_catalog.by_id("tiny")
        files = [{"path": "model-00000.safetensors", "size": 400_000}]
        ui = setup_flow.UI(False, stream=io.StringIO())
        with mock.patch.object(setup_download, "windows_curl", return_value="/mnt/c/curl.exe"), \
             mock.patch.object(setup_download, "probe_throughput", return_value=63e3), \
             mock.patch.object(setup_download, "probe_throughput_windows", return_value=3.3e6):
            self.assertEqual(setup_flow._maybe_windows_transport(ui, entry, files, None, "auto"),
                             "/mnt/c/curl.exe")
        with mock.patch.object(setup_download, "windows_curl", return_value="/mnt/c/curl.exe"), \
             mock.patch.object(setup_download, "probe_throughput", return_value=50e6), \
             mock.patch.object(setup_download, "probe_throughput_windows", return_value=40e6):
            self.assertIsNone(setup_flow._maybe_windows_transport(ui, entry, files, None, "auto"))
        with mock.patch.object(setup_download, "windows_curl", return_value=None):
            self.assertIsNone(setup_flow._maybe_windows_transport(ui, entry, files, None, "yes"))

    def test_existing_model_dir_skips_the_download(self):
        existing = os.path.join(self.tmp.name, "mine")
        os.makedirs(existing)
        Path(existing, "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe_text"}))
        Path(existing, "tokenizer.json").write_text("{}")
        code, text = self.run_setup(model_dir=existing, backend="cpu")
        self.assertEqual(code, 0, text)
        cfg = setup_flow.load_config()
        self.assertEqual(cfg["model_dir"], existing)
        self.assertEqual(cfg["backend"], "cpu")
        self.assertEqual(cfg["env"], {})
        self.assertFalse(any("/cdn/" in r["path"] for r in self.hub.requests))

    def test_the_menu(self):
        rows = setup_catalog.recommend(27 * 10**9, 800 * 10**9)
        out = io.StringIO()
        ui = setup_flow.UI(True, stream=out)
        with mock.patch("builtins.input", side_effect=["", "99", "2"]):
            first = setup_flow.choose_model(ui, rows)
            second = setup_flow.choose_model(ui, rows)
        self.assertEqual(first.id, "tiny")                       # Enter takes the recommendation
        self.assertEqual(second.id, [r for r in rows if r["fits"]][1]["entry"].id)
        menu = out.getvalue()
        self.assertIn("[recommended]", menu)
        self.assertIn(setup_catalog.FITS_EXPLAINED, menu)
        self.assertIn("type a number from 1", menu)

    def test_json_mode_prints_the_configuration(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = setup_flow.run(lambda a: setup_flow.cmd_setup(a, ui=setup_flow.UI(False, quiet=True)),
                                  setup_args(dir=self.models, pick="tiny", json=True))
        self.assertEqual(code, 0)
        result = json.loads(out.getvalue())
        self.assertEqual(result["config"]["backend"], "vulkan")
        self.assertEqual(result["config_path"], setup_flow.config_path())
        self.assertNotIn("status", result)                     # --no-start

    def test_no_compiler_and_no_prebuilt_says_what_to_install(self):
        tc = dict(TC_ALL, source_checkout=False, can_build=False, can_build_vulkan=False)
        os.remove(os.path.join(self.engines, "qwen36"))
        with mock.patch.object(setup_flow, "release_asset_suffix", return_value=None):
            with self.assertRaises(setup_flow.SetupError) as caught:
                setup_flow.resolve_engine(family_by_id("qwen36"), None,
                                          {"backend": "cpu", "missing": []}, tc, out=lambda *_: None)
        self.assertIn("no prebuilt engine", str(caught.exception))

    def test_list_json(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = setup_flow.run(setup_flow.cmd_setup, setup_args(list=True, json=True))
        self.assertEqual(code, 0)
        rows = json.loads(out.getvalue()[out.getvalue().index("["):])
        self.assertEqual(rows[0]["id"], "tiny")
        self.assertTrue(rows[0]["recommended"])


FAKE_SERVER = r'''
import http.server, json, os, signal, sys, tempfile
port = int(sys.argv[sys.argv.index("--port") + 1])
pidfile = os.path.join(tempfile.gettempdir(), f"coli-serve-{port}.pid")
with open(pidfile, "w") as f:
    f.write(f"{os.getpid()} fake\n")
with open(os.environ["FAKE_ENV_OUT"], "w") as f:
    json.dump({"argv": sys.argv[1:], "COLI_VULKAN": os.environ.get("COLI_VULKAN"),
               "EXTRA": os.environ.get("EXTRA")}, f)
PAGES = {"/health": {"status": "ok", "arch": "qwen36"},
         "/profile": {"seq": 1, "turns": [{"wall_s": 2.0, "completion_tokens": 10}]},
         "/v1/models": {"object": "list", "data": [{"id": "fake-model"}]}}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps(PAGES.get(self.path, {})).encode()
        self.send_response(200 if self.path in PAGES else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
def bye(*_):
    try: os.unlink(pidfile)
    except OSError: pass
    os._exit(0)
signal.signal(signal.SIGTERM, bye)
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


@unittest.skipUnless(os.name == "posix", "the stand-in server is stopped with SIGTERM")
class ServerControl(HomeTestCase):
    def setUp(self):
        super().setUp()
        self.launcher = os.path.join(self.tmp.name, "fake_coli.py")
        Path(self.launcher).write_text(FAKE_SERVER)
        self.env_out = os.path.join(self.tmp.name, "env.json")
        self.port = free_port()
        self.cfg = setup_flow.save_config({
            "status": "ready", "model": {"name": "Fake"}, "model_dir": self.tmp.name,
            "launcher": self.launcher, "engine": self.launcher, "backend": "vulkan",
            "env": {"COLI_VULKAN": "1", "EXTRA": "from-config"},
            "args": ["web", "--model", self.tmp.name, "--host", "127.0.0.1", "--port", str(self.port)],
            "host": "127.0.0.1", "port": self.port, "urls": setup_flow.urls("127.0.0.1", self.port)})
        patcher = mock.patch.dict(os.environ, {"FAKE_ENV_OUT": self.env_out, "EXTRA": "from-user"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.cleanup_server)

    def cleanup_server(self):
        pidfile = setup_flow.serve_pidfile(self.port)
        try:
            pid = int(Path(pidfile).read_text().split()[0])
            os.kill(pid, 15)
        except (OSError, ValueError):
            pass

    def wait_state(self, cfg, state, timeout=10):
        deadline = time.time() + timeout
        status = setup_flow.server_status(cfg)
        while status["state"] != state and time.time() < deadline:
            time.sleep(0.1)
            status = setup_flow.server_status(cfg)
        return status

    def test_start_status_stop(self):
        self.assertEqual(setup_flow.server_status(self.cfg)["state"], "stopped")
        status = setup_flow.start_server(self.cfg, background=True, open_browser=False,
                                         out=lambda *_: None, wait=10)
        self.assertIn(status["state"], ("ready", "loading"))
        status = self.wait_state(self.cfg, "ready")
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["model_id"], "fake-model")
        self.assertEqual(status["tokens_per_second"], 5.0)
        seen = json.loads(Path(self.env_out).read_text())
        self.assertEqual(seen["COLI_VULKAN"], "1")
        self.assertEqual(seen["EXTRA"], "from-user")             # the user's own environment wins
        self.assertIn("--no-browser", seen["argv"])
        self.assertTrue(os.path.exists(setup_flow.log_path("serve")))
        # Starting twice does not start a second server.
        again = setup_flow.start_server(self.cfg, background=True, open_browser=False,
                                        out=lambda *_: None)
        self.assertEqual(again["state"], "ready")
        report = setup_flow.status_report()
        self.assertEqual(report["server"]["state"], "ready")
        self.assertTrue(report["setup"]["ready"])
        result = setup_flow.stop_server(self.cfg, out=lambda *_: None)
        self.assertTrue(result["stopped"], result)
        self.assertEqual(self.wait_state(self.cfg, "stopped")["state"], "stopped")

    def test_a_taken_port_moves_to_the_next_free_one(self):
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", self.port))
        blocker.listen(1)
        self.addCleanup(blocker.close)
        with mock.patch.object(setup_flow, "_get_json", return_value=(None, None)):
            setup_flow.start_server(self.cfg, background=True, open_browser=False,
                                    out=lambda *_: None, wait=0.5)
        moved = setup_flow.load_config()
        self.assertNotEqual(moved["port"], self.port)
        self.assertEqual(moved["args"][moved["args"].index("--port") + 1], str(moved["port"]))
        self.port = moved["port"]                                 # cleanup stops this one


class CommandLine(HomeTestCase):
    def coli(self, *args):
        return subprocess.run([sys.executable, str(C_DIR / "coli"), *args], capture_output=True,
                              text=True, timeout=120, env=dict(os.environ, COLI_COLOR="0"))

    def test_status_and_logs_before_setup(self):
        result = self.coli("status")
        self.assertEqual(result.returncode, 1)
        self.assertIn("not set up yet", result.stdout)
        result = self.coli("status", "--json")
        self.assertEqual(json.loads(result.stdout)["configured"], False)
        self.assertEqual(self.coli("logs").returncode, 1)

    def test_setup_help_lists_the_options(self):
        result = self.coli("setup", "--help")
        self.assertEqual(result.returncode, 0)
        for flag in ("--yes", "--model", "--model-dir", "--dir", "--backend", "--no-gpu",
                     "--background", "--no-start", "--via-windows", "--json"):
            self.assertIn(flag, result.stdout)

    def test_start_before_setup_says_what_to_do(self):
        result = self.coli("start")
        self.assertEqual(result.returncode, 1)
        self.assertIn("coli setup", result.stderr)


if __name__ == "__main__":
    unittest.main()
