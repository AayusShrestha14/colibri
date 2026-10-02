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
A device that opens prints `[VK] <engine>: device ready, dense matrices on the
device` (or `on the CPU`), and why.

**Where the dense matrices go** is one rule, the backend's `coli_vk_dense_decide()`,
for every engine below. `COLI_VK_DENSE=1` puts them on the device, `COLI_VK_DENSE=0`
keeps them on the CPU. Unset, they go to the device, except where the device shares
the CPU's RAM (an integrated GPU, or a CPU device such as Lavapipe) and the engine
runs the routed-expert tier ([below](#the-routed-expert-tier-vk_tierc)): there they
stay on the CPU and the device takes the experts. On a Radeon 780M the dense matmuls,
one synchronous call each at the GPU's 800 MHz floor, cost more than the tier gained
(Qwen3.8 decode at 2.35 tok/s with them on the device, 3.80 without). Today that
case is qwen36 and qwen38; an engine that moves to the tier inherits it. A discrete
GPU keeps the dense matrices on the device by default. The GLM engine above reads
the same variable through the same function with its own default, off.

What these engines put on the device is their **resident** matrices, in the form
they already hold in RAM, uploaded at the first multiply (MiMo uploads them at
startup). Routed experts arrive from disk on every miss; qwen36 and qwen38 keep a
cache of them on the device with the shared expert tier
([below](#the-routed-expert-tier-vk_tierc)), MiMo an opt-in one of its own, the
others none yet.

| Engine | On the device | Weight formats | Stays on the CPU |
|---|---|---|---|
| qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B) | the dense trunk; routed experts on the expert tier | int8 rows; int4-g64 with `COLI_DENSE_BITS=4`; f32 with `COLI_DENSE_I8=0`; experts int4-g64, int4 per row, int8 per row or gs64 | DeltaNet `dn_a`/`dn_b`, vision tower, the experts the tier does not hold |
| qwen38 (Qwen3.8 Flash Next) | the trunk; routed experts on the expert tier | int8 trunk rows, bf16, f32 (`Q38_NATIVE_BF16=0`); experts int4-g64 (sidecar), FP8 128x128 blocks, bf16 | the MTP head's experts, the experts the tier does not hold |
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
  - qwen36's int8 dot (`COLI_DENSE_IDOT`, on by default) and its routed-expert
    kernel (`QWEN_EXPERT_ACT`, int8 by default): the expert tier's experts match
    `QWEN_EXPERT_ACT=f32`, to 2.5e-7 of the logits on the test fixture;
  - qwen38's int8 trunk;
  - qwenimage's `COLI_IMG_ACT8`.
- Two engines keep the CPU's exact arithmetic instead:
  - deepseek_v4 rounds activations to E4M3 on the host before the call, as its CPU
    kernel does.
  - inkling leaves its bf16 matrices on the CPU when the build has the AVX512-BF16
    dot (Zen 4/5, Sapphire Rapids), because that dot rounds activations to bf16.

**Memory.** The host copy stays as the CPU fallback. On an integrated GPU or APU,
which shares RAM with the CPU, the resident set is therefore held twice: size
`RAM_GB`/caps with that in mind. Freed tensors give their device memory back (see
the expert tier's section); MiMo's expert tier still never evicts by its own design.

**Status.** CI checks every engine above on Lavapipe (`tests/vulkan_engines.sh`, the
`vulkan-engines` job): each configuration gives the CPU run's tokens, and its matmul
count is above zero. That proves correctness, not speed.

The first real GPU measured is an integrated Radeon 780M (RADV) in a Ryzen 7 PRO
8700GE (16 threads, 61 GiB DDR5, NVMe): same binaries, cold page cache, load under 2.
The shader harness (`tests/vulkan_engines.sh shader`) passes every format case there.
The engines are correct on it and slower than the CPU today:

| Workload | CPU | Vulkan, 780M |
|---|---|---|
| Qwen3.8 Flash Next int4, decode 100 tokens | 3.55 tok/s | 1.53 tok/s |
| Qwen3.8 Flash Next, prefill 512 tokens | 51 s | 109 s |
| Qwen3.6-35B-A3B, decode | 5.97 tok/s | 3.06 tok/s (identical output) |
| Qwen3.6-35B-A3B, prefill 512 tokens | 39.8 s | 60.7 s |

Why, measured:
- **Decode**: every matmul is a synchronous submit, about 726 per token, and an
  integrated GPU reads the same RAM as the CPU.
- **Prefill**: the shader is a per-row GEMV, so each weight is read once per prompt
  row.

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

## The routed-expert tier (`vk_tier.c`)

The engines above keep their dense matrices on the device (on a discrete GPU; on
one that shares the CPU's RAM they stay on the CPU while this tier is on, see
[the other engines](#the-other-engines)); the routed experts are the other half of
a MoE model, and the one that does not fit. The tier keeps a cache
of them on the device the way a GPU-equipped PC should use its card:

- **What is resident adapts while you chat.** At startup the tier fills its budget
  from the expert history (`.coli_usage`, the hottest experts first, read from disk
  in parallel). After that, every expert the CPU computes is a candidate: it is
  promoted while there is room, or when it is hotter than the coldest resident by
  `tier.h`'s LFRU margin (25% + 4 routings), which is evicted. Heat is one per
  routing, halved every 1024 tokens; the history starts at 32 for a layer's hottest
  expert and in proportion below, so an expert of a new workload displaces the
  history's coldest residents after a few dozen routings of its own (it needs more
  than 1.25 x their heat + 4). A promotion copies the expert's bytes once on the engine thread
  (at most `COLI_VK_TIER_RATE` per token, 16) and an uploader thread writes it to
  the device; it serves from the next layer step on.
- **The device and the CPU compute at the same time.** For each MoE layer step the
  routed (row, expert) pairs whose expert is resident go to the device as ONE
  submit that nobody waits for: per expert, its rows run gate+up and the activation
  then down, all experts in one command buffer, on a queue of their own when the
  device has a second one (RADV's async compute, a second queue on NVIDIA and Intel),
  so the dense matmuls of the same layer do not wait behind it. Meanwhile the CPU
  loads and computes the other experts and the shared expert. Then the step joins.
- **Neither side waits for the other more than it must.** When a join keeps
  waiting (the device is the slower side: an integrated GPU at its floor clock),
  the tier hands the CPU the step's resident experts that the CPU also holds in RAM,
  beyond the device's share of the step's rows, and takes them back when the device
  finishes early; an expert only the device holds stays there (the CPU would read
  it from disk). A discrete card that finishes first keeps everything.
- **The sum does not depend on what was resident.** Every expert's output joins its
  row in routing (rank) order, the device's and the CPU's alike: the same order as a
  CPU-only run, so the device's experts differ from the CPU's only by their own
  summation order. A device row's bits do not depend on how many rows share its
  dispatch either (up to 15 rows an expert takes the per-row GEMV route), so an MTP
  verify's two rows get a decode step's bits. From 16 rows (prefill) an expert takes
  the tiled GEMM for gate, up and down.
- **Without `COLI_VULKAN`, nothing changes.** In a `VK=1` build with `COLI_VULKAN`
  unset, and in a build without `VK=1`, the stdout and the last logits of every
  qwen36 and qwen38 fixture configuration are the bytes of the build before the
  tier (110 configurations: bf16, FP8, int4-g64, int8, the MTP head, every prefill
  mode, both qwen36 expert kernels, the mixed container, four model geometries).

Engines on the tier today: **qwen36** (Qwen3.6, Qwen3-Coder, the 2.4T geometry: int8
per row or gs64, int4 per row or gs64 from either expert kernel, the mixed int4/int8
container) and **qwen38** (Qwen3.8 Flash Next: the int4-g64 sidecar, the release's
FP8 with 128x128 block scales, BF16). The MTP head's layer of qwen38 stays on the CPU
(its experts are FP8 beside an int4 sidecar). GLM-5.2's `COLI_VK_EXPERTS`, Kimi K3's
`K3_VK` and MiMo's `MIMO_VK_EXPERTS` are the older per-engine tiers; the others move
to this one in the next phase, see [Adding an engine](#adding-an-engine-to-the-tier).

With the CUDA expert tier built and on (`COLI_CUDA=1`) as well, **CUDA wins**: the
Vulkan tier stays off and says so (`[VK] tier <engine>: the CUDA expert tier is on
and wins`). The Vulkan dense trunk keeps running, on the device by default whatever
the device, since no Vulkan tier runs.

| Variable | Default | Effect |
|---|---|---|
| `COLI_VK_TIER` | on with `COLI_VULKAN=1` | `0`: no tier, the routed experts stay on the CPU (the dense trunk still uses the device). |
| `COLI_VK_TIER_GB` | measured | The tier's budget in GiB, within what the device can hold. Unset: below. |
| `COLI_VK_TIER_RESERVE_GB` | `1` | Device memory left to everything else (scratch, KV mirrors, the driver) on top of the dense weights the engine still has to place. |
| `COLI_VK_TIER_RATE` | `16` | Promotions per token at most (a prompt's forward gets this many per prompt token): each copies one expert on the engine thread. |
| `COLI_VK_TIER_BALANCE` | on | `0`: the device takes every resident expert of a step even when it is the slower side (see below). |
| `COLI_VK_TIER_WARM` | on | `0`: no warm start; the tier fills as experts pass by. |
| `COLI_VK_TIER_SYNC` | `0` | `1`: each layer step first waits for the uploads staged so far: residency then follows the routing alone (with `COLI_VK_TIER_BALANCE=0`, the run is reproducible). For tests and debugging. |
| `COLI_VK_TIER_GEMM_ROWS` | `16` | Rows from which an expert of a step takes the tiled GEMM instead of the per-row GEMV; `0` never. |
| `COLI_VK_TIER_QUEUE` | a second queue | `0`: the tier shares the main queue (its batches and the dense matmuls then serialize). |
| `COLI_VK_DENSE` | on, but off on a device sharing the CPU's RAM while the tier is on | `0`: the dense trunk stays on the CPU and the device takes the routed experts only; `1`: the trunk on the device whatever the device. Unset: on a discrete GPU, or with the tier off, on the device; on an integrated GPU or Lavapipe with the tier on, on the CPU. The startup line says which and why. (The GLM engine reads it through the same rule with its own default, off.) |
| `COLI_USAGE` | `<snap>/.coli_usage` | The history the warm start reads. qwen38 always keeps it; qwen36 keeps it only while the tier is on, and saves it at the end of every run and serve turn. |

**The budget.** On a discrete GPU: what `VK_EXT_memory_budget` says is free in
device-local memory, less the reserve and the dense weights the engine is about to
place there (Qwen3.8 puts 4.1 GiB of trunk on the device). On an integrated GPU (and on
Lavapipe), device memory IS the CPU's RAM: RADV on the Radeon 780M reports a 21 GiB
device-local heap and a 10.5 GiB host heap, together the 512 MiB carve-out and the
31 GiB of system RAM the kernel lets the GPU map; an allocation in either takes RAM
the CPU's cache and the page cache would otherwise have. There the default is a quarter of what
`MemAvailable` leaves once the engine's expert cache has grown to its configured
size (cap x layers x expert) and the dense weights are placed, less 2 GiB: the tier
never takes what that cache needs. `COLI_VK_TIER_GB` sets it explicitly. The startup
line says which rule applied:

```
[VK] tier qwen38: on, AMD Radeon 780M Graphics (RADV PHOENIX), budget 9.62 GiB = 3734 experts of 2.6 MiB (fmt 4 gs 64, down fmt 4 gs 64), shared RAM: a quarter of what the expert cache leaves, own queue, up to 16 promotions per token, balanced against the CPU
[VK] tier qwen38: warm start, 3734 experts from the history in 2.6s
```

**Memory that is given back.** Weight tensors are VkBuffers bound at offsets inside
256 MB device-memory blocks (one memory object per tensor makes every submit pay for
thousands of referenced allocations). `vk_alloc.h` hands those offsets out best-fit
and takes them back on free, coalesced; an emptied block goes back to the driver.
The tier's experts live in a pool of their own whose limit is the budget, at the
lower eviction priority (`VK_EXT_memory_priority`), so a pressed heap evicts experts,
never scratch or the dense trunk. A free while a batch may still read the tensor
waits for that batch's join.

**One line per run and serve turn** (stderr, beside the engine's own `[VK]` line):

```
[VK] tier qwen38 run: device 29464 of 59520 routed experts (49.5%; this run 29464 of 59520) | CPU RAM hits 18619, disk loads 8513 | resident 3734 (budget 3734, 9.61 GiB of 9.62 GiB, 39 blocks, frag 0.54) | uploads 4756 (10.89 GiB, 3734 warm), evictions 1022, skipped 0 queue + 2242 rate, failed 0 | device 8686.4 ms, CPU share 11136.6 ms, waited 3031.6 ms (65% of device time hidden) | balance: device share 0.05, 3869 rows handed to the CPU
```

- *device / routed*: where each routed (row, expert) pair ran; the rest is split by
  the engine's RAM cache into *RAM hits* and *disk loads*.
- *resident*, *budget*, the pool's blocks and fragmentation (`1 - largest free
  extent / free bytes`).
- *uploads* (warm-start ones included), *evictions*, promotions *skipped* because
  the upload queue was full or the per-forward rate was spent, uploads that
  *failed* (the device refused memory: the planned residency shrinks to what is
  there).
- *device*: the batches' device time (timestamps); *CPU share*: what the engine did
  between issue and join; *waited*: what the join then waited. The device time not
  waited for was hidden behind the CPU.
- *balance* (with the balancer on): the share of a step's resident rows the device
  keeps at the moment when the CPU holds them too, and the rows handed to the CPU
  since startup.

The dashboard's expert map (`EMAP`) shows a device-resident expert as tier 2 (VRAM),
the experts a device step served still light up in `HITS`, and qwen36's
`CACHE_ROUTE` ranks them like CUDA-resident ones.

### Measured on a Radeon 780M

Same box as above (Ryzen 7 PRO 8700GE, 16 threads, 61 GiB DDR5, NVMe, RADV), one
quiet run each after the model files were dropped from the page cache, the same
binary for every arm, `OMP_NUM_THREADS=8`. Qwen3.8 runs the int4-g64 sidecar at
cap 96, Qwen3.6 the int4 gs64 container at cap 64; decode is 100 tokens after a
25-token prompt (prompt included, as above), prefill a 512-token prompt
(`N_NEW=1`; Qwen3.8 with `Q38_PREFILL_BATCH_ROWS=512`).

Each arm set `COLI_VK_DENSE` explicitly; the trunk on the CPU is now this device's
default while the tier is on. Every tier arm starts from a history of one unrelated
conversation (a 231-token
prompt about planning a bakery's week, 100 tokens generated), as a user's would be;
*no warm start* starts from nothing. A first round, run from histories that had
seen the benchmark's own prompts, is at the end. Decode, 100 tokens (the rate the
engine reports; in brackets the whole process, load and warm start included):

| | Qwen3.8 Flash Next, int4 | Qwen3.6-35B-A3B |
|---|---|---|
| CPU | 3.51 tok/s (36.4 s) | 6.02 tok/s (23.6 s) |
| tier, trunk on the CPU (`COLI_VK_DENSE=0`) | 3.80 tok/s (36.9 s) | 8.03 tok/s (22.5 s) |
| the same, `COLI_VK_TIER_BALANCE=0` | 3.71 tok/s (37.6 s) | 7.86 tok/s (22.7 s) |
| the same, no warm start | 3.45 tok/s (37.0 s) | 6.10 tok/s (23.4 s) |
| tier and trunk on the device (`COLI_VK_DENSE=1`) | 2.35 tok/s (53.0 s) | 7.32 tok/s (23.7 s) |
| trunk on the device, no tier (`COLI_VK_TIER=0`, first round) | 1.60 tok/s (70.4 s) | 3.22 tok/s (38.1 s) |

Prefill of a 512-token prompt (time to the first token; in brackets the whole
process):

| | Qwen3.8 Flash Next, int4 | Qwen3.6-35B-A3B |
|---|---|---|
| CPU (first round) | 43.9 s (51.8 s) | 35.7 s (42.7 s) |
| tier, trunk on the CPU | 38.3 s (48.9 s) | 12.3 s (22.4 s) |
| the same, no warm start | 44.3 s (52.4 s) | 21.0 s (28.2 s) |
| tier and trunk on the device | 38.4 s (48.8 s) | 12.5 s (22.7 s) |
| trunk on the device, no tier (first round) | 43.2 s (51.2 s) | 37.1 s (44.1 s) |

What the numbers say, and what they do not:

- **The tier wins by the reads it saves.** At these caps the CPU's RAM cache misses
  often and every miss is an NVMe read; an expert on the device is neither read nor
  computed by the CPU. Qwen3.6's budget (12.5 GiB) holds 74% of its 10,240 experts
  and served 91% of the decode's routed pairs and 75% of the prefill's; Qwen3.8's
  (9.6 GiB) holds 15% of its 24,576 and served 50% of the decode's pairs and 16% of
  the prefill's. Hence 1.33x on Qwen3.6's decode and 2.9x on its time to the first
  token, against 1.08x and 1.15x on Qwen3.8.
- **The warm start is paid at startup.** Reading the history's experts took 2.3 to
  3.4 s (7.6 to 11.1 GiB) in every warm arm. The whole-process times include it, the
  rates do not: over a 100-token run Qwen3.8's tier comes out even with the CPU
  (36.9 s against 36.4 s), Qwen3.6's 1 s ahead. Without a history the tier fills as
  experts pass, at most 16 per token: not enough over 100 tokens for Qwen3.8 (3.45
  against 3.51 tok/s); Qwen3.6's prefill still gains (21.0 s against 35.7 s).
- **The trunk on the device costs more than the tier gains, here.** The trunk's
  synchronous matmuls at the 800 MHz floor (see above) make "trunk on the device, no
  tier" the slowest arm, and "tier and trunk" sits between it and the tier alone.
  Hence the default: on a device that shares the CPU's RAM, with the tier on, the
  trunk stays on the CPU (`COLI_VK_DENSE=1` puts it back). A discrete card keeps it
  on the device by default; nothing here measures one.
- **Overlap.** On Qwen3.8's decode the device computed 8.7 s of experts and the
  joins waited 3.0 s of it: 65% ran behind the CPU's share of the step. On Qwen3.6
  the device is the slower side (6.8 s of device time against 1.9 s of CPU share);
  the balancer moved the share to its floor, but the experts it holds are mostly not
  in the CPU's 64-slot cache, so there was little to hand back. With the balancer off
  the rates were 2% lower on both models, inside what one run to the next varies on
  this box.
- **The clock.** In the runs with the trunk on the CPU the GPU sat at its 800 MHz
  floor in 97 to 100% of the samples, except Qwen3.6's prefill (72% warm, 87%
  cold). Nothing here was run with the clock pinned.
- **Memory.** On this APU the tier's device memory is RAM, and it does not show in
  the process's RSS: the lowest `MemAvailable` during a decode fell from 52 to
  42 GiB (Qwen3.6) and from 40 to 31 GiB (Qwen3.8) with the tier on.
- **The text.** Qwen3.6 printed the CPU's text in every arm, decode and prefill.
  Qwen3.8 with the trunk on the CPU printed the CPU's 100 decode tokens in two of four
  runs; in the other two (balancer off, no warm start) the text left the CPU's at the
  72nd word. Its first token after the 512-token prompt was the CPU's in four of five
  runs. Its experts compute in f32 on both sides, so the device's and the CPU's
  differ only in summation order, about 1e-7 of the logits on the test fixtures. Over
  48 layers of top-10-of-512 routing and 512 tokens such differences flip routing
  near-ties and grow: after that prompt the logits differ from the CPU's by 0.36 on
  average (KL 0.10), and by as much between the tier's own two routes for prefill
  rows (the GEMM against the per-row GEMV: 0.34, KL 0.07). The trunk on the device
  moves them further (0.76, KL 0.49) through its int8 trunk (see Arithmetic above).
  A run with the tier is also not bit-reproducible by default: which experts a step
  finds resident depends on when the uploader finished, and the balancer on measured
  times (`COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0` removes both).

The first round (histories that had seen the benchmark's prompts: Qwen3.8's own
`.coli_usage` from earlier work, and for Qwen3.6 a run on the prefill prompt):
Qwen3.8 with the trunk on the CPU decoded at 3.70 tok/s with 56% of the pairs on the
device and reached the first token of the 512-token prompt in 32.9 s with 63% (the
second round's 38.3 s had 16%); Qwen3.6 decoded at 8.04 tok/s and prefilled in
12.8 s. A history that has seen the prompt helps Qwen3.8 and hardly matters for
Qwen3.6, whose budget holds most of its experts anyway.

### What was not measured

No discrete GPU was available. On one, the tier's experts sit in VRAM and the device
reads them at VRAM bandwidth, several times what the CPU gets from DDR; that is the
case the design is for, and nothing above is a prediction of it. What the 780M does
not have and a discrete card does: its own memory (here the device's experts and
the CPU's cache share the same DDR5 channels and the same 61 GiB), a clock that
leaves its floor under bursty work (the 780M's mostly did not), and PCIe uploads
(here an upload is a RAM copy). What the 780M does show is that the machinery
holds: the batches overlap the CPU, the history fills the budget in a few seconds,
eviction keeps the budget, and the fixtures' tokens are the CPU's.

### Integrated GPUs: reading the RAM cache in place

`VK_EXT_external_memory_host` lets the device read host memory where it is, with no
second copy: on an APU that would make the tier's experts the RAM cache's own slots.
The harness measures it (`COLI_VK_TEST_HOSTMEM=1 ./vk_test`): batches of 10
experts of Qwen3.8's shape (int4-g64, 2.76 MB each) cycling over 48 distinct
experts, once from the tier's device memory and once from page-aligned host memory
imported in place. On the 780M, three runs:

| | device time per batch |
|---|---|
| experts in the tier's device memory | 2.57 to 2.59 ms |
| experts in imported host memory | 3.24 to 3.77 ms (same bits) |

The copy the import would save costs 0.095 ms per expert into the tier's memory
(29 GB/s; 0.063 ms into ordinary memory), on the uploader thread, off the engine's
path. Reading in place is slower on every batch to save a copy paid once per
promotion, so the tier copies, on APUs too. Serving the RAM cache's own slots would
also need slots that are page-aligned, slots held against the cache's LRU while a
batch reads them, and slots laid out the way the shaders read them (today the copy
converts `expert_ffn.h`'s planar int4 and spreads Qwen3.8's FP8 block scales). The
backend enables the extension when the device has it; nothing outside the harness
uses it.

## The dense chain (`vk_chain.c`)

With the dense matrices on the device one `coli_vk_matmul` at a time, each matrix is
a submit and a host round trip: about 726 of them per Qwen3.8 decode token. On an
integrated GPU that made the dense part slower on the device than on the CPU (see
[the other engines](#the-other-engines)). The chain records a whole layer into one
command buffer instead, and keeps the residual stream on the device from one layer
to the next. qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B) and qwen38 (Qwen3.8 Flash
Next) run it: by default on a discrete GPU, and for qwen36 on an integrated one with
the expert tier (see [the default](#the-chain-on-a-radeon-780m)); `COLI_VK_CHAIN=1`
anywhere.

**What runs where, per layer** (S rows: one at decode, a prompt chunk at prefill):

| | qwen36 | qwen38 adds |
|---|---|---|
| device, frame A1 | the previous layer's MoE output joining the residual (in the CPU's order: routed experts in rank order, then the gated shared expert, then the add); the input RMSNorm; the gated attention (q/k/v, per-head q/k norm, RoPE, the new K/V rows into the device cache, attention with the output gate, o_proj) or the Gated DeltaNet (qkv/z/b/a, the causal convolution with its ring, the recurrence with its state, the gated norm, out_proj); the residual add; the post-attention norm; the router logits | the four hyper-connection streams: each block's gated-residual read (per-stream norm, the low-rank pair, the stream mix, the inject weights) and its write-back; the PLE layer's projections, gate and dilated convolution (the n-gram table rows come up from the host); QSA with its indexer: each block's pooled key computed once on the device when a step completes it, and the top-k selection per query row |
| host | the router's softmax and top-k; the routed experts (the expert tier's device batch and the CPU's share, joined in rank order); the new K/V rows copied into the host's cache | the new index-key rows too |
| device, frame A2 (not waited for) | the shared expert and its gate, while the host computes the routed experts | |
| last frame | the final norm and lm_head on the last row | the final mixer; lm_head on the last one or two rows (an MTP verify) |

Per layer the host gets the normalized rows the routed experts read (D floats a row)
and the router logits (E floats a row), and the new K/V (and index-key) rows of an
attention layer; it sends the routed sum back (D floats a row). A model without
routed experts (Qwen3.8-27B) has no host step: its whole forward is one frame.

**The state, and who owns it.**
- The residual stream: on the device for the whole forward.
- The attention KV cache (and Qwen3.8's index keys): the host's copy stays canonical,
  as with the GLM engine's KV mirror. The device holds a mirror per layer with a
  watermark: rows below it equal the host's. A step from `pos_base` first uploads
  the rows between the watermark and `pos_base`; a step that runs on the CPU lowers
  the watermark to its `pos_base`; a cache that grows is mirrored again. Qwen3.8's
  pooled block keys have a watermark of their own, lowered to the first block a step
  rewrites (a rejected draft's block is recomputed when a step completes it again).
- The DeltaNet recurrent state and conv rings, and Qwen3.8's PLE ring: on the device
  while the chain runs (60 MB on Qwen3.6-35B, too much to copy per token). The host's
  copy is brought back before anything reads it there (a pinned snapshot, the prompt
  cache, a CPU step) and pushed up after anything writes it there (a reset: a fill
  with zeros on the device; a restored snapshot: an upload).
- An MTP verify (S = 2) snapshots the DeltaNet states, the conv rings and the PLE ring
  after its first row on the device (the shaders write the snapshot as they pass
  that row); a rejected draft swaps the device buffers, as the CPU swaps its own.
  Its matrices take the per-row GEMV, so its rows get a decode step's bits.

Prompt-cache and prefix reuse need nothing else: a reused prefix is rows below the
watermark and a recurrent state that already sits where the next step expects it.
The serve and prefix tests run with the chain on (below).

**What stays on the CPU.** The routed experts the tier does not hold, the router's
top-k, the embedding gather and the vision tower's rows, Qwen3.8's n-gram table reads
and the MTP head (its experts are FP8 beside the int4 sidecar, two rows per draft).
The chain declines, and the per-matrix path runs with the state synced first, under
the CUDA expert tier (CUDA keeps its priority), a qpack container, PILOT prefetch, or
a geometry outside its shaders (head dim above 256, a DeltaNet value head above 128 or
key head above 256, a conv kernel above 9). A device lost while the chain holds the
recurrent state does not stop the engine: it rebuilds that state on the CPU from the
prefix record (the ids the state was built from; the KV rows are the host's already),
a prefill's worth of CPU work, and runs on the CPU from there. Only a state the ids do
not describe (a turn with an image) cannot be rebuilt: that stops the engine with a
message. `COLI_VK_CHAIN_FAULT=n` fakes the loss at the n-th frame (the tests use it).

**The shaders** (`shaders/chain_*.comp`, each documented at its top): RMSNorm over
segments (rows, heads with a gate between them, streams with a weight slice each,
zero-centred or not, L2); RoPE from a host table (the CPU's own cosf/sinf at the CPU's
angles, so M-RoPE is just another table); grouped-query attention with an online
softmax over tiles of 128 positions, the output gate and an optional selection list;
the DeltaNet convolution and the recurrence (one workgroup per value head, a column of
the state in registers per thread, the gated norm fused); the element-wise steps; the
QSA block keys and selection; the PLE gate and convolution; and a decode GEMV for the
trunk's formats (int8 rows, int4-g64, bf16, f32) that reads 16 bytes per lane per step
and spreads a row over a cluster of lanes. The matrices of a prefill chunk take the
backend's fp32 tiled GEMM from its threshold (S ≥ 2 and S·O ≥ 4096). Activations are
f32 throughout, as the CPU's f32 path.

| Variable | Default | Effect |
|---|---|---|
| `COLI_VK_CHAIN` | on for a discrete GPU; on an integrated GPU with the expert tier, what the engine measured (qwen36 on, qwen38 off); off on a CPU device | `1`: every layer's dense chain on the device; `2`: prompts only (forwards of more than two rows; decode and MTP verifies on the per-matrix path, the state moving between the two); `0`: the per-matrix path. The `[VK] <engine>: dense chain ...` line says which and why. |
| `COLI_VK_CHAIN_ROWS` | `512` | Prompt rows per chunk: a longer prompt runs every layer chunk by chunk (the device's scratch is sized for one chunk). |
| `COLI_VK_CHAIN_GEMV` | on | `0`: the decode matrices take `qmatmul.comp`'s GEMV instead of `chain_gemv.comp`'s. |
| `COLI_VK_CHAIN_SPIN_US` | `2000` | How long a wait on a chain frame polls the fence before blocking. |
| `COLI_VK_CHAIN_PROF` | off | `1`: one `[VK] chain profile` line of device time per kind of op (timestamps). |

Each run and serve turn prints `[VK] <engine> chain: N forwards, F frames (ops,
matmuls, tiled GEMM), the time spent waiting for the device, the routed experts' host
time and the device memory the chain holds`.

### The chain on a Radeon 780M

The box and the method of [the tier's measurements](#measured-on-a-radeon-780m): Ryzen
7 PRO 8700GE, RADV, `OMP_NUM_THREADS=8`, every run after the model files were dropped
from the page cache, 1-min load under 2, every tier arm from the same history of one
unrelated conversation, the same binary for every arm; Qwen3.8 runs the int4-g64
sidecar at cap 96, Qwen3.6 the int4 gs64 container at cap 64. Decode is 100 tokens
after a 25-token prompt (the rate the engine reports; in brackets the whole process),
prefill a 512-token prompt (`N_NEW=1`; Qwen3.8 with `Q38_PREFILL_BATCH_ROWS=512`).
Where a cell lists two or three numbers, they are separate rounds.

| Decode | Qwen3.8 Flash Next, int4 | Qwen3.6-35B-A3B |
|---|---|---|
| CPU | 3.49 tok/s (36.5 s) | 6.01 tok/s (23.6 s) |
| tier, trunk on the CPU | 3.81, 3.81, 3.84 tok/s (36.8 s) | 8.03, 8.06 tok/s (22.5 s) |
| tier and chain (`COLI_VK_CHAIN=1`) | 3.18, 3.18 tok/s (41.7 s) | 9.94, 9.92, 9.97 tok/s (20.2 s) |
| tier and chain on prompts only (`COLI_VK_CHAIN=2`) | 3.64 tok/s | |

| Prefill, 512 tokens | Qwen3.8 Flash Next, int4 | Qwen3.6-35B-A3B |
|---|---|---|
| CPU | 43.6 s (51.5 s) | 35.7 s (42.8 s) |
| tier, trunk on the CPU | 38.7, 38.6 s (49.4 s) | 12.2 s (22.3 s) |
| tier and chain (prompts: `COLI_VK_CHAIN=1` or `2`) | 30.1, 30.1, 30.1 s (40.5 s) | 9.5, 9.5 s (19.7 s) |

What the numbers say:

- **Qwen3.6: the chain wins both.** Decode 24% over the tier alone, 65% over the CPU;
  the first token 22% sooner than the tier alone. A decode token is 82 frames (two a
  layer and the head) where the trunk on the device took about 290 synchronous
  submits; the host waited 53 ms a token for the device's frames and spent 40 on the
  routed experts (the tier's batch, the CPU's share, their join).
- **Qwen3.8: prefill wins, decode loses.** The first token comes 22% sooner, but
  decode is 17% slower than the tier alone. Its trunk is 3.6 G weights in the layers
  and 0.6 G in lm_head, int8 rows, read whole every token (more than its routed
  experts); the device's decode GEMV, at the 800 MHz floor the GPU mostly sits at,
  reads int8 at 30 to 43 GB/s back to back, where the CPU's integer kernel takes the
  trunk in about 84 ms a token (73 ms of resident matmuls and 11 of lm_head in the tier
  arm's timers), about 50 GB/s. `COLI_VK_CHAIN_PROF=1` put 77% of the chain's device
  time in those GEMVs. Running only the prompts on the device (`COLI_VK_CHAIN=2`) keeps the prefill
  gain and still loses 5% of decode: on shared RAM the trunk's device copy (1.1 GiB of
  the budget here) is taken from the tier.
- **The clock.** With the trunk on the CPU the GPU sat at its 800 MHz floor in 97 to
  100% of the samples during decode. The chain keeps it busy enough to leave the floor
  part of the time: 53% of the samples at 800 MHz on Qwen3.6's decode, 39% on
  Qwen3.8's, 60% and 84% on the prefills. Nothing here pinned the clock.
- **The text.** Qwen3.6's chain printed the CPU's 100 decode tokens word for word (the
  tier alone left them at the 22nd word in this round); Qwen3.8's left them at the
  59th word of 78 (the tier alone at the 43rd). The chain multiplies f32 activations
  where the CPU's int8 kernels round them (see Arithmetic above); after the 512-token
  prompts every arm gave the CPU's first token.
- **Memory.** The chain holds 160 to 290 MiB on the device for Qwen3.6 and 350 to
  720 MiB for Qwen3.8 (state, mirrors and scratch for a 512-row chunk), beside the
  trunk's device copy (1.9 GiB of int8 for Qwen3.6). On an integrated GPU both come
  out of the tier's budget: 12.05 instead of 12.48 GiB on Qwen3.6, 8.4 instead of
  9.6 GiB on Qwen3.8.

**The default** (`coli_vk_chain_decide`, next to `coli_vk_dense_decide`): on a discrete
GPU the chain is on. On an integrated GPU with the expert tier on, each engine passes
what it measured here: qwen36 on, qwen38 off (decode, a chat's steady state, is slower
in both modes; `COLI_VK_CHAIN=2` is the choice for long prompts). Without the tier, and
on a CPU device such as Lavapipe, it is off. The startup line says which and why:

```
[VK] qwen36: dense chain on (an integrated GPU with the expert tier: measured faster on decode and prefill; COLI_VK_CHAIN=0 off, 1 on, 2 prompts only)
```

**What was not measured.** No discrete GPU was available. On one the trunk sits in
VRAM, read at several times the CPU's bandwidth, the clock is not held at a floor
between bursts, and the host round trip per layer crosses PCIe (a few KB a row each
way); that is the case the chain's default is set for, and nothing above is a
prediction of it. Not timed either: contexts past 2048 tokens (where Qwen3.8's QSA
selects blocks instead of attending to all of them), serve sessions, Qwen3-Coder and
Qwen3.8-27B (no checkpoints on the box); their correctness is the Lavapipe gates'.

### Adding an engine to the chain

The recipe qwen36_chain.h and qwen38_chain.h follow, for the engines still on the
per-matrix path:

1. **Parameters and tensors once.** Pack every norm weight and small parameter vector
   into one device buffer (`vkc_buf` + one `vkc_write`) and pass offsets; resolve every
   matrix to the device copy the per-matrix path already uploads (`coli_vk_tensor_ensure`
   into the engine's own `vk` field), so the two paths never hold a matrix twice.
2. **Own the state explicitly.** Attention caches: keep the host's canonical, copy each
   step's new rows back (a few KB a layer), mirror on the device behind a watermark
   lowered by every CPU write. Recurrent state too large to copy per token: on the
   device, with a "who holds the newest copy" flag synced at every host read (snapshots,
   prompt caches) and every host write (resets, restores). A speculative verify writes a
   snapshot of the recurrent state at its first row in the shader that walks the rows,
   and a rejection swaps buffers.
3. **One frame per layer up to the first thing the host must decide** (the router's
   top-k for the routed experts), the CPU-independent tail (the shared expert) in a
   frame nobody waits for, and the join (routed sum + shared, in the CPU's order) at the
   head of the next layer's frame.
4. **The CPU's arithmetic order wherever it is cheap to keep** (the conv sum order, the
   MoE combine, the RoPE angles from a host table), f32 activations, and a gate per
   configuration in `tests/vulkan_engines.sh` against the CPU's tokens and logits.

What each remaining architecture needs on top of today's shaders:

| Engine | Attention / mixer | New pieces |
|---|---|---|
| colibri.c (GLM-5.2), glm53, deepseek_v41, kimi_k3 (MLA layers) | MLA: q_a/kv_a, the latent norms, q_b, RoPE on the rope dims, a latent + rope cache | the absorb core (`attention_absorb.comp`) already reads a device KV mirror with GLM's watermark (`vk_kv_valid`): record it as a chain op with the o-projection fused (the universal layout's eight bindings hold its seven); DSA's index keys and top-k are `chain_qsa.comp`'s selection with blocks of one position, and the absorb shader takes the selection list as `chain_attn.comp` does; GLM's MTP layer stays on the CPU like qwen38's head, or runs in the chain (MLA has no recurrent state: a rejected draft is a lowered watermark) |
| deepseek_v4 | MLA with compressed (CSA) and hierarchical (HCA) KV, mHC | the manifold hyper-connections are qwen38's stream read/apply with a Sinkhorn normalization (an element-wise op that iterates); the compressors' rolling windows are rings like the conv's, snapshotted the same way; the CPU rounds activations to E4M3 before its fp8 matmuls, so the chain needs that rounding as an element-wise op to keep the same arithmetic |
| kimi_k3 (KDA layers) | Kimi Delta Attention: a gated delta rule whose decay is a vector over the key channels | `chain_dnrec.comp` with the decay per key row (one `exp(g_k)` per row of the column, loaded beside q and k in shared memory) instead of one per head; the short convolution is `chain_dnconv.comp`; its output gate and norm as the gated norm |
| mimo | sliding-window attention (and full layers) | `chain_attn.comp` with a window (positions from `max(0, pos - W + 1)`: one more push constant), the cache optionally a ring of W rows (the host mirror then indexes `t % W`) |
| inkling | grouped attention, MoE with a shared expert | qwen36's attention and combine as they are; its per-position heads are matmuls of the residual rows |
| olmoe | attention with q/k norm, MoE without a shared expert | qwen36's Qwen3-Coder geometry (all attention, no gate, no shared expert) is the same chain |

## Adding an engine to the tier

The integration steps are in [`c/vk_tier.h`](../c/vk_tier.h); in short:

1. **Describe the experts** (`VktConfig`): geometry, how RAM holds gate/up and down
   (`VktSrc`: int8 per row or grouped, the int8 copy of an int4 container, int4
   pairs signed or `v+8`, `expert_ffn.h`'s planar int4-g64, int3-g64, MXFP4 with f32
   or ue8m0 scales, fp8 per group or in square blocks, bf16, f32), the activation
   (`VKT_ACT_SWIGLU` with an optional clamp, `VKT_ACT_SITU`), the most assignments a
   step carries, the RAM the expert cache may still take and the dense bytes still to
   come to the device. `vkt_init(&cfg, rt_counts_all())` after the device and the
   history; then `atexit(coli_vk_shutdown)` and `atexit(vkt_shutdown)`, in that
   order, so that at exit the tier lets go of its experts and the device is
   destroyed before the drivers unload.
2. **Warm start** (optional): `vkt_plan`, read each planned expert into a buffer of
   the loader's own (any number of threads), `vkt_put`, then `vkt_put_done`.
3. **Every MoE step**: `vkt_issue(layer, x, S, K, idx, taken)`; compute the pairs not
   taken on the CPU into rows of their own, and `vkt_note` every expert whose bytes
   are in RAM; the shared expert; `vkt_join` (when the issue took any); then add
   every rank of every row in order, the device's row where `taken`. A failed join
   (device lost) leaves the taken pairs to the CPU and turns the tier off.
4. **Report**: `vkt_report("run"|"turn", ram_hits, disk_loads)` beside the engine's
   `[VK]` line; `vkt_resident(l, e)` gives EMAP its tier 2.

What each remaining engine needs, from reading its code:

| Engine | Experts in RAM (`VktSrc`) | Activation | Where | To watch |
|---|---|---|---|---|
| colibri.c (GLM-5.2) | int4 per row `I4U_PAIRS_ROW`, int4-gs `I4U_PAIRS_GS`, int3-g64 `I3_G64`; gate, up, down separate | SwiGLU | `moe()`'s Vulkan block replaces the `COLI_VK_EXPERTS` registry | sums per expert in union order today: move to rank order; fmt 6 (E8/IQ3, rotated input) stays on the CPU; the MTP layer is int8 (another format); the block now serves only S <= 4 |
| glm53 | int4 gs64 `I4U_PAIRS_GS` 64 | SwiGLU with `swiglu_limit` | `ffn_layer` | its CPU clamps even at limit 0 (no guard), the shader treats 0 as no clamp: pass the config's value and check `L > 0` on the CPU side first; shared expert written first |
| inkling | one `gate_up` tensor [2I, D], rows 0..I-1 gate: pass `g = p13`, `u = p13 + I rows`; int4 per row `I4U_PAIRS_ROW` or int8 `I8_ROW` | SwiGLU | `moe` | routed and shared weights normalized together, `route_scale x rgs` already in the weights |
| kimi_k3 | MXFP4 `MXFP4_E8M0` 32, gate `w1`, up `w3`, down `w2` | SiTU-GLU, a = 4, b = 25 (`VKT_ACT_SITU`) | `moe_forward` / `expert_apply` (experts in the latent space) | replaces `K3_VK` (synchronous, never evicts); the CPU's `K3_IDOT` rounds activations to int8 |
| deepseek_v41 | MXFP4 `MXFP4_E8M0` 32 | SwiGLU with limit | `moe_run_at` (already sums per (row, rank) in rank order) | DSpark stages have caches of their own |
| deepseek_v4 | FP4 + ue8m0/32 = `MXFP4_E8M0` 32; pinned experts are repacked rows16 | SwiGLU with limit, **plus** bf16 rounding of gate/up, the route weight applied before down and bf16 rounding of the output | `moe_token_pipeline` (ascending expert id), `v4_moe_batch_union` | needs an activation variant with those roundings and the host's E4M3 rounding of x (as its fp8 dense path does); undo rows16 or keep pinned experts off the tier; hash-routed layers |
| mimo | MXFP4 `MXFP4_E8M0` 32, stored down, ds, gate, gs, up, us | SwiGLU | `moe` | replaces `MIMO_VK_EXPERTS` (synchronous, never evicts) |
| olmoe | int8 per row `I8_ROW`, one merged tensor g, u, d | SwiGLU | `moe` | history only with `COLI_USAGE` |

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
- The expert tier: `tests/test_vk_alloc` (in `make check`) runs the sub-allocator
  against a byte map, 40,000 random steps included; `make vk-tier-check VK=1` runs
  `vk_tier.c` against a CPU reference for every expert source format, the warm
  start, adaptation with eviction and partial batches; the harness runs the expert
  batch for every weight format and both activations, a row's bits checked
  independent of the batch, and the tier pool's budget with frees while a batch is
  in flight. `tests/vulkan_engines.sh qwen` gives the CPU's tokens with the tier on
  in every qwen36 and qwen38 expert format, under eviction, with MTP and with the
  trunk on the CPU; `qwen-sanitize` runs the same under ASan and UBSan.
- The dense chain: `make vk-chain-check VK=1` (`tests/test_vk_chain.c`) runs every chain
  op against a CPU reference; `tests/vulkan_engines.sh qwen-chain` gives the CPU's tokens
  with the chain in every qwen36 geometry and expert container and every qwen38 format,
  the last logits within 1e-4 of the largest one where both sides use f32 activations
  (measured 2e-7 and below on the fixtures), prefill in chunks, an image, MTP drafts
  rejected, accepted and alternating, a device lost mid-run, the qwen38 oracle targets,
  the prefix-reuse contract and serve sessions frame for frame; `qwen-chain-sanitize`
  runs the chain under ASan and UBSan.
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

- GLM-5.2 (this section's engine): decode-focused, its `COLI_VK_EXPERTS` tier and the
  attention core serve `S<=4`; prefill uses the CPU/batched paths (dense projections
  do run on VK at prefill). The shared expert tier serves prefill too; GLM moves to
  it in the next phase.
- The expert tier's uploads are host writes into host-visible device memory: a
  discrete card needs Resizable BAR for them (above). A staging copy on a transfer
  queue, for cards without it, is not written.
- DSA top-k selection, ragged multi-slot serving, and quantized-KV caches
  fall back to the CPU attention path.
- Not yet done: a fully resident-layer pipeline for the engines other than qwen36 and
  qwen38 ([the dense chain](#the-dense-chain-vk_chainc) is theirs), Polaris/gfx803 validation on real
  hardware (the shaders use dynamic subgroup sizes and are wave64-safe by
  construction). The cooperative-matrix GEMM is measured on RDNA3 only.
