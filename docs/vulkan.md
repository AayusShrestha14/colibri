# Vulkan backend (any GPU with a Vulkan 1.2 driver)

colibrì includes an opt-in Vulkan compute backend that runs the whole GLM
decode compute path on any GPU a Vulkan driver can see — no CUDA, no ROCm.
That includes cards the vendor stacks have dropped (ROCm 7 removed Polaris:
an RX 580 runs here via RADV) and, measured on an RX 9070 (RDNA4), it is
*faster* than the ROCm/HIP backend on the same card.

```bash
cd c
make glm VK=1                # needs libvulkan + glslc (shaderc) for the shaders
COLI_VULKAN=1 COLI_VK_DENSE=1 COLI_VK_ATTN=1 \
PIN=<model>/.coli_usage PIN_GB=0 COLI_NO_OMP_TUNE=1 \
./coli run "Hello" --topp 0.7
```

Requirements: `libvulkan` and a Vulkan **1.2** ICD with
`GL_KHR_shader_subgroup_arithmetic` (any Mesa RADV, AMDVLK, NVIDIA or Intel
ANV driver from the last several years), plus `glslc` at build time. The
backend picks the most capable physical device (discrete > integrated) and
degrades to the CPU path on any failure — a wedged GPU can slow a run, never
corrupt it.

Set `COLI_NO_OMP_TUNE=1` on multi-core boxes: the engine's OMP self-tune
(active spin-wait) is skipped under `COLI_CUDA`/`COLI_METAL` but not under
Vulkan, and spinning worker threads starve the async I/O pool (measured
CPU expert bandwidth 28 → 5 GB/s without it).

**Discrete cards need Resizable BAR.** The weight tiers allocate
HOST_VISIBLE|DEVICE_LOCAL memory; with ReBAR disabled that combination only
exists in a ~256 MB BAR window, and the driver silently places everything
beyond it in system RAM — the tier then *reports* resident experts while every
access crosses PCIe, slower than the CPU path (measured 0.11 vs 0.24 tok/s
either side of the BIOS toggle on an RX 9070 XT). The engine now warns at init
when the host-visible slice of VRAM is small; if you see that warning, enable
Resizable BAR / Smart Access Memory in the BIOS. Unified-memory APUs are
unaffected.

The compiled shaders are found via `COLI_VK_SHADERS` (either the
`qmatmul.spv` file or the directory holding the `.spv` set); unset, the
engine looks in `shaders/` next to the binary, then relative to the CWD.

### Windows (MSYS2)

In the MSYS2 **UCRT64** shell ([quickstart.md](quickstart.md)), add the Vulkan
headers, the loader's import library and `glslc`, then build:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-vulkan-headers \
  mingw-w64-ucrt-x86_64-vulkan-loader mingw-w64-ucrt-x86_64-shaderc
cd c
make colibri.exe VK=1
```

The binary is statically linked like the default Windows build, plus one
import: `vulkan-1.dll`, the loader the GPU driver installs in `System32`, so it
runs outside MSYS2 with nothing added to `PATH`. The next-to-the-binary shader
lookup above is Linux-only: on Windows run from `c\` or set `COLI_VK_SHADERS`.
To check the driver before downloading a model, point `SNAP` at a folder
holding only a `config.json`, as the CI's Lavapipe job does; the backend
initialises before any weight is read:

```powershell
# in c\
New-Item -ItemType Directory -Force $env:TEMP\vkprobe | Out-Null
'{"model_type":"glm_moe_dsa"}' | Set-Content $env:TEMP\vkprobe\config.json
$env:SNAP = "$env:TEMP\vkprobe"; $env:COLI_VULKAN = "1"; $env:COLI_NO_OMP_TUNE = "1"
.\colibri.exe    # prints "[VK] ready: <GPU>", then exits: there is no model
```

## What runs on the GPU

| Piece | Env | Mechanism |
|---|---|---|
| Routed experts (hot set) | `COLI_VK_EXPERTS=N` (default 320) | Top-N experts by `.coli_usage` heat uploaded **once at startup** into a VRAM registry; at decode they are served from VRAM with **no RAM slot, no disk read, no prefetch**, as one async fused batch (gate+up+silu→down, hidden on-device) overlapped with the CPU computing the remaining experts. Shown as the `vk` bucket in the hit-rate line. |
| Dense projections | `COLI_VK_DENSE=1` | q_a+kv_a fused into one submit, q_b, o; shared expert as a single fused expert-group submit. Resident int4/int8 weights upload once. |
| MLA attention core | `COLI_VK_ATTN=1` | One dispatch per layer: absorbed query, scores over the KV window, softmax, weighted latent, value rows, **fused with the o-projection** (the context vector never leaves the GPU). The latent/rope KV lives in a persistent per-layer device mirror, appended ~2.3 KB/token/layer with the same invalidation points as the CUDA KV shadow. |

The `PIN_GB=0` (with `PIN` still set) in the example is deliberate: the VRAM
registry holds the same hot experts a RAM pin would, so the pin's RAM is
better spent on the adaptive LRU cache. Keep `PIN` set so AUTOPIN does not
re-pin from history.

## The other engines

Every engine links the same backend in a `VK=1` build (`make <engine> VK=1`;
`make deepseek-v4 VK=1` for DeepSeek V4) and opens it with `COLI_VULKAN=1` once its
weights are loaded. Kimi K3 has its own expert tier (`K3_VK`, see
[ENVIRONMENT.md](ENVIRONMENT.md)) and glm53 its own section in
[glm53-flash.md](glm53-flash.md). For the engines below, a missing device or missing
shaders prints `[VK] <engine>: no usable Vulkan device ..., running on the CPU` and
the run continues on the CPU. That differs from the GLM engine above, which exits.

What these engines put on the device is their **resident** matrices, in the form
they already hold in RAM, uploaded at the first multiply (MiMo uploads them at
startup). Routed experts arrive from disk on every miss and stay on the CPU, with
one opt-in exception for MiMo.

| Engine | On the device | Weight formats | Stays on the CPU |
|---|---|---|---|
| qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B) | the dense trunk | int8 rows; int4-g64 with `COLI_DENSE_BITS=4`; f32 with `COLI_DENSE_I8=0` | DeltaNet `dn_a`/`dn_b`, vision tower, routed experts |
| qwen38 (Qwen3.8 Flash Next) | the trunk | int8 trunk rows, bf16, f32 (`Q38_NATIVE_BF16=0`) | routed FP8 experts |
| inkling | dense and shared-expert matrices | int8 and int4-g64 (dense-int4g64 container), f32, bf16 | routed experts, embedding and audio lookups, CUDA residents; bf16 on CPUs with the AVX512-BF16 dot (see below) |
| olmoe | attention q/k/v/o, router, lm_head | f32 | routed experts, embedding |
| deepseek_v41 | the trunk, vision included | fp8 in 32x32 ue8m0 tiles, bf16 | routed experts |
| deepseek_v4 | resident dense layers, head, router, compressors | fp8 in 128x128 blocks, bf16 | routed experts, the indexer's `weights_proj`, DSpark stages, the `--oracle` path |
| mimo | trunk and vision tower; up to `MIMO_VK_EXPERTS=N` routed experts | native fp8/bf16, int8, f32 (`MIMO_DENSE_BITS`); experts as MXFP4 | router |
| qwenimage | the DiT's matrices | int8, bf16, f32 (`COLI_IMG_BITS`) | text encoder, VAE, attention |

Each engine ends a run, and each serve turn, with
`[VK] <engine>: N matmuls on the GPU`. That count is how you tell a path that ran
from one that only initialised.

**Arithmetic.** The device reads the same weights the CPU reads and multiplies
them by f32 activations.
- Where the CPU's default kernel also uses f32 activations, the two differ only in
  the order of the sums.
- Where the CPU kernel rounds activations first, the device result instead matches
  the CPU's f32-activation setting, so tokens can drift from the CPU default after
  a few steps. These kernels are:
  - qwen36's int8 dot (`COLI_DENSE_IDOT`, on by default);
  - qwen38's int8 trunk;
  - qwenimage's `COLI_IMG_ACT8`.
- Two engines keep the CPU's exact arithmetic instead:
  - deepseek_v4 rounds activations to E4M3 on the host before the call, as its CPU
    kernel does.
  - inkling leaves its bf16 matrices on the CPU when the build has the AVX512-BF16
    dot (Zen 4/5, Sapphire Rapids), because that dot rounds activations to bf16.

**Memory.** The host copy stays as the CPU fallback. On an integrated GPU or APU,
which shares RAM with the CPU, the resident set is therefore held twice: size
`RAM_GB`/caps with that in mind. The weight arena does not return freed tensors'
memory, which is also why MiMo's expert tier never evicts.

**Status.** CI checks every engine above on Lavapipe (`tests/vulkan_engines.sh`, the
`vulkan-engines` job): each configuration gives the CPU run's tokens, and its matmul
count is above zero. That proves correctness, not speed. None of these engines has
been measured on a real GPU yet.

## Prefill: the tiled GEMMs

`qmatmul.comp` is a GEMV per activation row: at S rows every weight is fetched and
decoded S times, which holds a Radeon 780M at 80 GFLOP/s whatever S is. From S = 2,
`coli_vk_matmul` (the resident-matrix path of every engine) takes a tiled GEMM
instead, so the API and the callers are unchanged:

- `qmatmul_gemm.comp`, fp32, every format: a workgroup owns a BM-output x BN-row tile
  of y, decodes its weight rows into shared memory once per 32-input step and runs
  them against BN activation rows staged beside them, a TM x TN block per thread.
  Group scales fold into the decoded weight; each step sums into a fresh partial
  (blocked summation), which keeps the error at the GEMV's level.
- `qmatmul_coop.comp`, where the device has `VK_KHR_cooperative_matrix` with a
  16x16x16 fp16 x fp16 -> fp32 subgroup shape and a settable subgroup size: the
  formats whose weights decode exactly to fp16 (int8, int4, int3-g64, MXFP4, fp8;
  grouped ones with gs % 32 == 0). Nothing is rounded to fp16: each activation row is
  scaled by a power of two the host computes during the upload, split into an fp16
  high part (top 11 bits) and an fp16 low part, and the MMA runs on both; every
  16-input product then joins fp32 accumulators, times the group scale. The harness
  holds it to the fp32 GEMM's bound.
- Each shader is built at a few tile widths (BN 32 for a 32-row prefill chunk, 64,
  128); a call takes the narrowest that covers its S. The threshold, measured on the
  780M: S >= 2 and S*O >= 4096 (a 48-output matrix stays on the GEMV up to S = 32).
  `COLI_VK_GEMM_MIN_S` overrides it, `COLI_VK_COOP=0` keeps the fp32 GEMM.

Measured on a Radeon 780M (RDNA3, RADV, Mesa 26.0) sharing DDR5 with a Ryzen 7 PRO
8700GE, `OMP_NUM_THREADS=8`. One matrix, I = 2560, O = 6144, S = 512, back to back
(the harness, `COLI_VK_TEST_GEMM_BENCH=1`; the CPU column is the kernel an engine
runs for that storage, `-march=native`):

| fmt | GEMV | tiled GEMM (fp32) | cooperative matrix | CPU, 8 threads |
|---|---|---|---|---|
| 1 int8 | 82 GFLOP/s | 1431 | 1909 | 841 (int8 activations, VNNI) |
| 2 int4 | 104 | 1486 | 2164 | 224 |
| 4 int4-g64 | 86 | 1406 | 2026 | 488 (int8 activations, VNNI) |
| 5 int3-g64 | 86 | 1441 | 2039 | 125 |
| 7 MXFP4 | 64 | 1390 | 1928 | 199 |
| 10 f32 | 29 | 948 | | 224 |
| 11 bf16 | 51 | 1222 | | 25 |
| 12 fp8 | 43 | 1259 | 1735 | 33 |

End to end, prefill of 512 tokens (`N_NEW=1`, cold page cache, same binaries):

| | CPU | Vulkan, GEMV | Vulkan, tiled GEMM |
|---|---|---|---|
| Qwen3.8 Flash Next (int8 trunk, bf16) | 50.7 s | 111.1 s | 53.8 s |
| same, `Q38_PREFILL_BATCH_ROWS=512` | 44.7 s | | 44.3 s |
| Qwen3.6-35B-A3B (int8 dense) | 40.9 s | 62.0 s | 56.3 s |

On this APU the GEMM brings Qwen3.8's Vulkan prefill from 111 s to the CPU's
level: level with it with 512-row prefill chunks (44.3 against 44.7 s), 6% behind
at the default 32-row chunks; the generated token is the CPU's on both models.
`VK_PROF=1` shows where the time goes. Qwen3.8 at the default chunks spends 8.4 s in
resident matmuls on the device against the CPU's 6.5 s: 6.8 s in 5,052
cooperative-matrix GEMMs (the DeltaNet projections at S = 32, 3.1 s; shared experts
and router, 1.7 s; attention and gated residuals at S = 512, 2.0 s) and 1.6 s in
2,273 GEMVs (1,024 one-token PLE projections, 0.86 s; matrices too narrow for the
GEMM, 0.7 s). Two things measured there hold it at parity:
- A clock stuck at 800 MHz. The engines call the device synchronously, one matrix
  at a time between CPU phases, and with `power_dpm_force_performance_level=auto`
  the GPU stays at its 800 MHz floor through those millisecond bursts (over 97% of
  the samples during the Qwen3.8 runs): a GEMM that takes 1.1 ms back to back takes
  2.3 ms after a 3 ms CPU gap. Fewer, larger calls (512-row chunks) are what lift
  it to parity.
- Per-token calls. Qwen3.6 without the CUDA tier projected its DeltaNet inputs one
  token at a time (`deltanet()` in qwen36.c): 46,081 of its 46,281 device matmuls
  were S = 1 and stayed on the GEMV (20 s of its 56).

The engines now project per block of rows wherever prefill used to call the device
once per token, recurrences and gathers still consuming the rows in order, and with
CPU outputs byte-identical to the per-token build: Qwen3.6's DeltaNet inputs and
out_proj, Qwen3.8's PLE keys and values, DeepSeek V4.1's router, indexer, index keys,
compressor and vision tower, DeepSeek V4's router, compressors and index queries,
GLM-5.3's device-resident projections, Kimi K3's DSA index keys and Inkling's
per-position heads. Same 780M, same runs:

| | CPU | Vulkan, tiled GEMM | device matmuls (GEMV / GEMM) |
|---|---|---|---|
| Qwen3.6-35B-A3B, per token | 40.9 s | 56.3 s | 46,081 / 200 |
| Qwen3.6-35B-A3B, per block | 36.0 s | 37.2 s | 1 / 380 |
| Qwen3.8 Flash Next, per token | 50.7 s | 53.8 s | 2,273 / 5,054 |
| Qwen3.8 Flash Next, per block | 52.8 s | 55.3 s | 1,249 / 5,086 |

The CPU gains too where the block lets a matrix stay in cache across rows (Qwen3.6's
DeltaNet: 7.1 to 2.5 s). Qwen3.8's block saves its 0.9 s of PLE GEMVs, inside the
run-to-run spread of its expert reads (cold page cache, about 2 s).

## Correctness

- `gcc -O3 -DVK_TEST backend_vulkan.c -o test_vk -lvulkan -lm && ./test_vk
  shaders/qmatmul.spv` runs a CPU-reference exactness harness over every
  primitive (GEMV int4/int8 across shapes incl. the long-row o-projection,
  fused gate+up, the full expert group sync and async, the matmul pair, and
  the absorb attention core incl. causal S=2, kv_start windows, int8, and
  long-context cases), and both tiled GEMMs for every weight format at S = 16, 64
  and 512 with odd I and O, tail groups and odd group sizes; a GEMM case fails if the
  call did not take the GEMM it names. Typical maxrel ~1e-5..2e-3 (fp32 reduction
  order). `COLI_VK_TEST_MATMUL_ONLY=1` stops after the GEMV and GEMM format cases.
- Engine-level: greedy decode with the full stack matches the pure-CPU
  engine token-for-token on the validation prompt.
- int4 weights decode as offset-binary (nibble−8), byte-identical layout to
  the CPU path — no repacking.
- Khronos validation layers: the backend never enables them, so the loader
  does. `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` turns on the core
  checks; add `VK_LAYER_VALIDATE_SYNC=true` for synchronization validation
  (or point `VK_LAYER_SETTINGS_PATH` at a directory holding a file named
  exactly `vk_layer_settings.txt`). The harness above reports no hazards
  under it. Known layer defect, SDK 1.4.357.1 on MoltenVK: submit-time
  synchronization validation segfaults inside the layer at `vkDeviceWaitIdle`
  during shutdown; set `VK_LAYER_SYNCVAL_SUBMIT_TIME_VALIDATION=false`, or
  read stdout through a pty, since the crash lands in an `atexit` handler
  before stdio flushes.

## Measured performance (AMD RX 9070, RDNA4, RADV/Mesa 26.1)

Expert-MLP primitive (K experts, int4 6144→2048→6144, per-call incl. readback):
Vulkan **0.11–0.13 ms/expert** vs the production ROCm/HIP expert group
**0.179 ms/expert** — ~35% faster. The decode MLA attention core runs 3.7×
faster than the HIP kernel on the same card. End-to-end GLM-5.2 (744B int4,
NVMe-streamed) decode on a 12-core Zen2 + RX 9070 box: Vulkan
**1.7–1.8 tok/s** (64-token) / **1.6** (256-token) / **1.58 sustained**
(512-token) vs the HIP backend at 1.5–1.55 on identical settings.
The two write-combined-memory rules that make this possible: buffers the CPU
reads back must be HOST_CACHED (ReBAR VRAM reads at ~40 MB/s otherwise), and
everything else lives HOST_VISIBLE|DEVICE_LOCAL.

## Benchmarking against other backends

Two defaults will silently skew any Vulkan-vs-CUDA/HIP comparison:

- **MTP speculation**: CUDA/HIP builds disable model drafts by default
  (`DRAFT` auto-resolves to 0 under `COLI_CUDA=1`, see #163), while CPU and
  Vulkan runs keep `DRAFT=3`. The arms then execute different decode loops —
  the speculative arm routes ~2× the expert positions per emitted token
  (rejected draft positions still pay their expert I/O), which dominates on
  storage-bound boxes. Output is identical either way (greedy verify is
  lossless), so nothing looks wrong. Pin `DRAFT=0` (or `DRAFT=3
  COLI_CUDA_MTP=1`) explicitly on **both** arms.
- **GPU clocks**: decode dispatches are microsecond bursts that never ramp
  DPM on their own; the memory clock can sit parked through an entire run.
  Pin `power_dpm_force_performance_level=high` (both arms) or disclose it.

Also note `experts loaded/token` in the run stats counts *routed positions*
(including rejected speculative ones) before any cache/tier is consulted —
it does not fall when the VK tier serves a hit; the `vk` bucket in the
hit-rate line is the tier-effectiveness number.

## Limits and future work

- Decode-focused: the expert tier and attention core serve `S<=4`; prefill
  uses the CPU/batched paths (dense projections do run on VK at prefill).
- DSA top-k selection, ragged multi-slot serving, and quantized-KV caches
  fall back to the CPU attention path.
- Not yet done: a fully resident-layer pipeline, Polaris/gfx803 validation on real
  hardware (the shaders use dynamic subgroup sizes and are wave64-safe by
  construction). The cooperative-matrix GEMM is measured on RDNA3 only.
