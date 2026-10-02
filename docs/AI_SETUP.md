# Setting up colibri with an AI coding assistant

This page is written for an AI coding assistant that a user asked to install,
start, check or stop colibri on their machine. Every step is a command with a
machine-readable result. A person can follow it too; the one-step path for
people is `START-HERE.bat` (Windows) or `./start-here.sh` (Linux, macOS), see
[quickstart.md](quickstart.md).

If your client speaks the Model Context Protocol, the same operations are
available as tools: see [MCP_SERVER.md](MCP_SERVER.md).

## Ground rules

- **Ask before downloading.** Models are 19 GB to 1.6 TB. Show the user the
  recommended model, its size and where it will go, and get a yes.
- **Ask before installing system packages.** The setup never runs `sudo`,
  `apt`, `pacman` or `winget` itself. When it needs a package it prints the
  exact command; show it to the user and run it only with their consent.
- **Reruns are safe.** Every step is idempotent. If a command stops (network,
  timeout, Ctrl+C), run the same command again: it continues where it stopped.
- **Prefer the Linux disk on WSL.** Never put the model under `/mnt/c` (or any
  `/mnt/<drive>`): the engine streams the model from disk, and that path is
  many times slower. The default, `~/colibri-models`, is right.

## 0. Prerequisites

Python 3.10 or newer and a copy of the repository.

```bash
python3 --version                              # Windows: py -3 --version
git clone https://github.com/JustVugg/colibri  # or use the folder the user already has
cd colibri
```

On Windows run the commands below from PowerShell or cmd in the repository
folder, with `py -3 c\coli` where this page writes `python3 c/coli`. If Python
is missing on Windows, `winget install -e --id Python.Python.3.12 --scope user`
installs it for the current user (ask first).

A C compiler is optional. With one (gcc and make on Linux, Xcode tools on
macOS, MSYS2 UCRT64 on Windows) the engine is built for this machine, with the
GPU when one is usable. Without one, the prebuilt engine from the GitHub
release is used, which runs on the CPU.

## 1. Look at the machine

```bash
python3 c/setup_hw.py --json             # RAM, disk, CPU features, GPUs
python3 c/coli setup --list --json       # every model against this machine
```

`setup_hw.py --json` reports `memory.total` and `memory.available` (bytes),
`disk.free_bytes` for `~/colibri-models` (pass another folder as an argument),
`cpu.features`, `gpu.vulkan` (the Vulkan device the engine would use, with its
`type`, `integrated` or `discrete`, and `budget_bytes` or `device_local_bytes`),
`gpu.vulkan_icd` (a driver manifest found under the home folder, typical on
WSL) and `gpu.nvidia` (from nvidia-smi).

`setup --list --json` returns one object per model:

| Field | Meaning |
|---|---|
| `id` | what `--model` takes |
| `download_gb` | size of the download |
| `ram_min_gb`, `ram_good_gb` | below the first it does not run; from the second it runs as measured |
| `dense_gb` | the part that always stays in RAM |
| `runs` | `from RAM` (small models) or `streams experts from the SSD` (large ones) |
| `status` | `good`, `tight`, `needs-ram`, `needs-disk` or `unsupported` |
| `fits`, `reason` | whether it can be installed here, and why |
| `recommended` | exactly one model is the default |

"Fits" means: the dense part, which always stays in RAM, plus a minimum
expert cache fit in RAM, and the download fits on the disk.

Tell the user the recommendation and the alternatives that fit, with their
sizes. Large models stream from the SSD and are much slower than the small
ones that run from RAM.

## 2. Install

Non-interactive, with the model the user chose:

```bash
python3 c/coli setup --yes --model qwen36-35b --no-start
```

What it does, in order: detects the hardware, picks the engine build (CUDA
for an NVIDIA card when the CUDA toolkit is installed and the engine has a
CUDA path, else Vulkan when a Vulkan GPU answered and the Vulkan headers and
`glslc` are installed, else the CPU), builds or fetches the engine, downloads
the model with resume, runs the planner on the files, and writes the run
configuration. It prints each step.

The download can take hours. If your command runner has a time limit, start
it detached and poll the status instead of waiting:

```bash
# Linux, macOS, WSL
nohup python3 c/coli setup --yes --model qwen36-35b --no-start > colibri-setup.log 2>&1 &
```

```powershell
# Windows
Start-Process -WindowStyle Hidden py -ArgumentList '-3','c\coli','setup','--yes','--model','qwen36-35b','--no-start' `
  -RedirectStandardOutput colibri-setup.log -RedirectStandardError colibri-setup.err
```

Then poll every minute or so:

```bash
python3 c/coli status --json
```

`install.phase` moves through `build`, `download`, `ready` (configured) and
`started` (server launched); `error` and `interrupted` end it early, and
`install.running` says whether the install process is still alive. During the
download `install.done` and `install.total` are bytes and `install.rate` is
bytes per second. On `error`, `install.message` says why; fix the cause and
rerun the same setup command.

### The setup options

| Option | Effect |
|---|---|
| `--yes`, `-y` | no questions: the recommendation and every default |
| `--model ID` | the model to install (ids from `--list`) |
| `--model-dir DIR` | use a model already on disk instead of downloading |
| `--dir DIR` | where models are downloaded (default `~/colibri-models`) |
| `--backend auto\|cpu\|vulkan\|cuda` | force the engine build; `--no-gpu` is `--backend cpu` |
| `--host H`, `--port N` | where the server listens (default `127.0.0.1:8000`) |
| `--no-start` | stop after writing the configuration |
| `--background` | start the server detached at the end |
| `--no-browser` | do not open a browser tab |
| `--reconfigure` | choose again even if a setup is complete |
| `--list`, `--all` | show the catalog against this machine and exit |
| `--via-windows auto\|yes\|no` | WSL only: download through Windows' `curl.exe` |
| `--no-verify` | skip the checksums of downloaded files |
| `--json` | machine-readable result (implies `--yes` and a background start) |

### When a GPU package is missing

If the machine has a GPU that the build cannot use yet, the setup continues
on the CPU and prints a line like:

```
  to use the GPU through VULKAN, first run:  sudo apt install libvulkan-dev glslc mesa-vulkan-drivers
```

Show that command to the user. If they run it (or let you run it), run
`python3 c/coli setup --yes --no-start` again: the setup remembers that a GPU
was waiting for packages, sees that they are there, rebuilds the engine for
the GPU and rewrites the configuration. The download is not repeated.

On Windows the GPU build needs MSYS2 with its UCRT64 packages; the line names
them. The prebuilt Windows engine runs on the CPU.

### WSL

The setup measures the download speed from WSL and from Windows with a short
ranged request when the download is larger than a few GB. If Windows is much
faster (it can be: 63 KB/s against 3.3 MB/s was measured on one machine), it
downloads through Windows' `curl.exe` into the same Linux folder. Force it with
`--via-windows yes`, or disable it with `--via-windows no`.

A Vulkan driver installed by hand under the home folder (Mesa's Dozen driver is
the usual one on WSL) is found and written into the run configuration as
`VK_ICD_FILENAMES`.

## 3. Start

```bash
python3 c/coli start --background --no-browser
python3 c/coli status --json
```

Wait until `server.state` is `ready` (`loading` first; loading takes from
seconds to minutes depending on the model). Then give the user:

| Field in `status --json` | Use |
|---|---|
| `server.urls.browser` | the dashboard, for the user |
| `server.urls.openai_base_url` | `.../v1`, for OpenAI-compatible clients |
| `server.urls.anthropic_base_url` | for Anthropic-compatible clients (they append `/v1/messages`) |
| `server.model_id` | the `model` value API requests must send |
| `server.tokens_per_second` | speed of the last answer, prompt time included |

A quick check that it answers:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "<server.model_id>", "messages": [{"role": "user", "content": "Hello"}]}'
```

`python3 c/coli start` without `--background` runs in the foreground and opens
the browser: that is what a person wants, not a tool call.

If the configured port is taken by another program, the server moves to the
next free port and the configuration is updated; read the URLs from
`status`.

## 4. Check, read the logs, stop

```bash
python3 c/coli status          # human-readable; --json for the fields above
python3 c/coli logs -n 50      # server log of a background start
python3 c/coli logs --install  # log of a detached install
python3 c/coli stop            # stops the configured server and its engine
```

`coli stop` without `--port` stops the server the setup configured. Rerunning
`python3 c/coli setup` later starts the configured server directly (in the
foreground; add `--background --no-browser` for a detached start), with no
download and no build.

## Files

Everything the setup writes lives in one folder: `~/.local/share/colibri` on
Linux, `~/Library/Application Support/colibri` on macOS,
`%LOCALAPPDATA%\colibri` on Windows. `COLI_SETUP_HOME` overrides it.

| File | Content |
|---|---|
| `setup.json` | the run configuration: model, engine, backend, environment, launcher arguments, URLs |
| `install-state.json` | phase and progress of the current or last install |
| `logs/serve.log`, `logs/install.log`, `logs/build.log` | logs |
| `runtime/` | a prebuilt release, when there is no compiler |

The model folder holds `.colibri-download.json`, which lists the files and
says when the download is complete. Partial files end in `.part`.

`HF_TOKEN` (or the token `hf auth login` saved) is used for gated repositories.
`HF_ENDPOINT` points the downloads at a mirror.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `download stopped: ...` | rerun the same command; it resumes from the bytes on disk |
| `checksum mismatch` | the partial file was removed; rerun |
| `needs N GB free for the download` | pass `--dir` with a folder on a bigger disk |
| `the published release ... predates <model>` | there is no compiler and the last release cannot run that model: install the compiler (the message names the command) or pick another model |
| `Already set up: ...` but the user wants another model | `--reconfigure`, or `--model ID` |
| `server.state` stays `loading` | `coli logs -n 50`; large models take minutes to load |
| `plan warning: RAM budget cannot hold one expert slot` | the model does not fit in the free RAM: close programs or choose a smaller model |
