#!/usr/bin/env bash
# Every engine's Vulkan path against its own CPU run, on Lavapipe (Mesa's software
# Vulkan), one family per call so CI can run them side by side:
#
#   bash tests/vulkan_engines.sh qwen | qwen-sanitize | inkling-olmoe | mimo-qwenimage | deepseek
#   bash tests/vulkan_engines.sh kimi | kimi-mimo-sanitize
#   bash tests/vulkan_engines.sh shader    # the qmatmul formats, the expert batch and the tier, no engine
#
# Needs libvulkan-dev, glslc and mesa-vulkan-drivers, plus the Python packages of
# the family's tiny fixtures (see the vulkan-engines job in .github/workflows/ci.yml).
#
# Lavapipe is a CPU rasteriser: nothing here says anything about speed. What it does
# prove is that every resident format an engine uploads reaches the shader and comes
# back as the CPU computes it. Each configuration is gated on two things:
#   - the Vulkan run gives the tokens of the CPU run with the same snapshot and
#     settings (and, where the engine has one, passes its own oracle);
#   - its "[VK] <engine>: N matmuls on the GPU" line has N > 0, because a hook that
#     declines every matrix would otherwise pass the first gate trivially. A
#     configuration of the routed-expert tier with the dense trunk on the CPU (the
#     default on Lavapipe while the tier is on, see dense_where) gates on its count of
#     routed experts the device served instead.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}
export COLI_NO_OMP_TUNE=1
PY=${PY:-python3}

fail() { echo "FAIL: $*"; exit 1; }

# vk_count <engine> <log>: N from the last "[VK] <engine>: N matmuls on the GPU" line
vk_count() {
  local n
  n=$(sed -n "s/^\[VK\] $1: \([0-9][0-9]*\) matmuls on the GPU.*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}
need_gpu() {  # <engine> <log> <tag>
  [ "$(vk_count "$1" "$2")" -gt 0 ] || { cat "$2"; fail "$3: no matmul ran on the device"; }
}
same_tokens() {  # <cpu log> <vk log> <tag>: the engines' "C engine" token lines
  grep -a '^C engine' "$1" > cpu.tok || true
  grep -a '^C engine' "$2" > vk.tok || true
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; fail "$3: Vulkan tokens differ from the CPU"; }
}

# The shader itself: every weight format against a CPU reference, before any engine;
# then the expert batch and the weight pool in the same harness, and the routed-expert
# tier (vk_tier.c) on a synthetic model in every source format.
shader_formats() {
  cc -O2 -pthread -DVK_TEST backend_vulkan.c -o vk_test -lvulkan -lm
  COLI_VK_TEST_MATMUL_ONLY=1 ./vk_test shaders/qmatmul.spv | tee vk_test.log
  tail -1 vk_test.log | grep -qx PASS || fail "qmatmul format cases"
  make tests/test_vk_tier VK=1
  ./tests/test_vk_tier shaders/qmatmul.spv | tee vk_tier.log
  tail -1 vk_tier.log | grep -qx PASS || fail "routed-expert tier"
}

# tier_count <engine> <log>: N from the last "[VK] tier <engine> run: device N of M" line;
# tier_evictions <engine> <log>: the evictions of that line
tier_count() {
  local n
  n=$(sed -n "s/^\[VK\] tier $1 run: device \([0-9][0-9]*\) of .*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}
tier_evictions() {
  local n
  n=$(sed -n "s/^\[VK\] tier $1 run: .* evictions \([0-9][0-9]*\),.*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}

# dense_where <engine> <log> <tag> <env...>: where the run put the dense trunk, checked
# against what it asked for. "[VK] <engine>: device ready, dense matrices on the
# device|CPU" must agree with the matmul count; COLI_VK_DENSE=1 must put the trunk on
# the device; with COLI_VK_DENSE unset and the tier on, Lavapipe (a CPU device, its
# memory the CPU's RAM) must keep it on the CPU, as an integrated GPU does.
dense_where() {
  local eng=$1 log=$2 tag=$3 where; shift 3
  if grep -qa "^\[VK\] $eng: device ready, dense matrices on the device" "$log"; then
    need_gpu "$eng" "$log" "$tag"; where=device
  else
    [ "$(vk_count "$eng" "$log")" = 0 ] || { cat "$log"; fail "$tag: dense matmuls ran on the device with the trunk on the CPU"; }
    where=CPU
  fi
  case " $* " in
    *" COLI_VK_DENSE=1 "*) [ $where = device ] || { cat "$log"; fail "$tag: COLI_VK_DENSE=1 left the trunk on the CPU"; } ;;
    *" COLI_VK_DENSE="*) ;;
    *) if grep -qa '^\[VK\] ready: llvmpipe' "$log" && grep -qa "^\[VK\] tier $eng: on" "$log"; then
         [ $where = CPU ] || { cat "$log"; fail "$tag: Lavapipe with the tier on kept the trunk on the device"; }
       fi ;;
  esac
  echo $where
}

# tier_gate <engine> <tag> <env...> -- <argv...>
# The routed-expert tier (vk_tier.c) against the CPU: the same tokens as the CPU run
# with the same settings, the device served some of the routed experts, and the dense
# trunk ran where it was asked to (dense_where). With EVICT=1 the run must also have
# evicted (its budget is set below the hot set). COLI_USAGE points at a fresh file: no
# warm start from an earlier run's history. COLI_VK_TIER_SYNC=1: a fixture's whole run
# can end before the uploader thread is first scheduled (it did in 5 of 40 runs with
# the trunk on the CPU), so the gate waits for each staged upload at the next step.
tier_gate() {
  local eng=$1 tag=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f tier.usage
  env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" COLI_USAGE=tier.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 ./"$eng" "$@" > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "$tag"
  [ "$(tier_count "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: no routed expert ran on the device"; }
  if [ "${EVICT:-0}" = 1 ]; then
    [ "$(tier_evictions "$eng" vk.log)" -gt 0 ] || { grep '\[VK\] tier' vk.log; fail "$tag: the budget forced no eviction"; }
  fi
  local where; where=$(dense_where "$eng" vk.log "$tag" "${envs[@]}") || { echo "$where"; exit 1; }
  echo "OK $tag: tokens = CPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.log | tail -1), $(grep -a -o 'evictions [0-9]*' vk.log | tail -1), trunk on the $where"
}

# vk_gate <engine> <placed-regex> <tag> <env...> -- <argv...>
# CPU arm and Vulkan arm of one configuration; both must exit 0 (the engine's own
# oracle), give the same tokens, run matmuls on the device, and report the expected
# placement ("placed int8 a, int4 b, f32 c" / "int8 a, bf16 b, f32 c").
vk_gate() {
  local eng=$1 placed=$2 tag=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || { cat cpu.log; fail "$tag: CPU run"; }
  env "${envs[@]}" COLI_VULKAN=1 ./"$eng" "$@" > vk.log 2>&1 || { cat vk.log; fail "$tag: Vulkan run misses the oracle"; }
  same_tokens cpu.log vk.log "$tag"
  need_gpu "$eng" vk.log "$tag"
  grep -qE "\[VK\] $eng: .*placed .*$placed" vk.log || { grep '\[VK\]' vk.log; fail "$tag: expected placement '$placed'"; }
  echo "OK $tag: $(grep -a -o 'Matching tokens: [0-9/]*' vk.log), tokens = CPU, $(grep -o '[0-9]* matmuls on the GPU.*' vk.log | tail -1)"
}

family_qwen() {
  make qwen36 qwen38 VK=1
  # qwen36: the hybrid, Qwen3-Coder (qwen3_moe), the 27B dense and the 2.4T geometry
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen3-coder-30b --out qwen3_coder_tiny --ref-mode full --emit-ref qwen3_coder_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen38-27b-dense --out qwen38_27b_tiny --ref-mode full --emit-ref qwen38_27b_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen38-2p4t --seed 3 --out qwen38_2p4t_tiny --ref-mode full --emit-ref qwen38_2p4t_tiny/ref_full.json
  local fx cap caps
  # These arms test the dense trunk's formats: COLI_VK_DENSE=1, because on Lavapipe (a
  # CPU device sharing the CPU's RAM) the trunk otherwise stays on the CPU while the
  # expert tier is on; the tier runs beside it.
  for fx in qwen36_tiny qwen3_coder_tiny qwen38_27b_tiny qwen38_2p4t_tiny; do
    $PY tools/convert_qwen36.py --model $fx --out ${fx}_c --ebits 8
    # cap=1 evicts on every routed expert; on the 92-layer 2.4T geometry that costs
    # ~90 s on Lavapipe for nothing the device path adds, so that one runs at cap=8.
    caps="1 8"; [ $fx = qwen38_2p4t_tiny ] && caps=8
    for cap in $caps; do
      # the CPU job's own configuration: f32 dense weights (fmt 10), token-exact
      # against transformers and equal to the CPU run
      vk_gate qwen36 'f32 [1-9]' "qwen36 $fx f32 cap=$cap" COLI_VK_DENSE=1 COLI_DENSE_I8=0 SNAP=${fx}_c -- $cap 8 $fx/ref_full.json
    done
    # int8 dense rows (fmt 1). int8 weights alone miss the torch oracle on some
    # fixtures, CPU or GPU alike, so this arm gates on the CPU's tokens only, with
    # COLI_DENSE_IDOT=0 giving the CPU the shader's f32 activations.
    COLI_DENSE_IDOT=0 SNAP=${fx}_c ./qwen36 8 8 $fx/ref_full.json > cpu.log 2>&1 || true
    COLI_DENSE_IDOT=0 COLI_VK_DENSE=1 COLI_VULKAN=1 SNAP=${fx}_c ./qwen36 8 8 $fx/ref_full.json > vk.log 2>&1 || true
    same_tokens cpu.log vk.log "qwen36 $fx int8"
    grep -qE '\[VK\] qwen36: [1-9][0-9]* matmuls on the GPU.*placed int8 [1-9]' vk.log || { cat vk.log; fail "qwen36 $fx int8: nothing placed"; }
    echo "OK qwen36 $fx int8: tokens = CPU, $(grep -o '[0-9]* matmuls on the GPU.*' vk.log | tail -1)"
  done

  # The routed-expert tier on every expert container qwen36 reads (COLI_VULKAN=1 turns
  # it on; the runs above already had it): int8 per row (fmt 1), the shared kernel's
  # planar int4-g64 (fmt 4), int4 per row from the unpacked slots (fmt 2), int8 gs64
  # (fmt 13) and the mixed int4 gate/up + int8 down container, at cap=1 with the trunk
  # where the default puts it (on the CPU here) and at cap=8 with the trunk on the
  # device (COLI_VK_DENSE=1); the tier alone (COLI_VK_DENSE=0); and a budget of two
  # experts, which must evict as the routing moves.
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny64 --ref-mode full --inter 64 --emit-ref qwen36_tiny64/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_c --ebits 4 --gs 64
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_g8 --ebits 8 --gs 64
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_d8 --ebits 4 --gs 64 --down-bits 8
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_i4r --ebits 4
  local D
  for cap in 1 8; do
    D=; [ $cap = 8 ] && D=COLI_VK_DENSE=1
    tier_gate qwen36 "qwen36 tier int8 cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- $cap 8 qwen36_tiny/ref_full.json
    tier_gate qwen36 "qwen36 tier int4-g64 planar cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny64_c -- $cap 4 qwen36_tiny64/ref_full.json
    tier_gate qwen36 "qwen36 tier int4 per row cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny_i4r -- $cap 4 qwen36_tiny/ref_full.json
    tier_gate qwen36 "qwen36 tier int8 gs64 cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny64_g8 -- $cap 8 qwen36_tiny64/ref_full.json
    tier_gate qwen36 "qwen36 tier mixed int4/int8 cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny64_d8 -- $cap 4 qwen36_tiny64/ref_full.json
  done
  tier_gate qwen36 "qwen36 tier alone (COLI_VK_DENSE=0)" COLI_VK_DENSE=0 COLI_DENSE_I8=0 SNAP=qwen36_tiny64_c -- 8 4 qwen36_tiny64/ref_full.json
  EVICT=1 tier_gate qwen36 "qwen36 tier, a budget of two experts" COLI_VK_TIER_GB=0.00002 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_full.json
  # COLI_VK_TIER=0: the dense trunk alone, as before the tier; with no tier the default
  # puts the trunk on the device, Lavapipe included
  COLI_VK_TIER=0 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_full.json > cpu.log 2>&1 || true
  COLI_VK_TIER=0 COLI_VULKAN=1 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_full.json > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "qwen36 COLI_VK_TIER=0"
  ! grep -q '^\[VK\] tier' vk.log || { cat vk.log; fail "qwen36 COLI_VK_TIER=0: the tier started"; }
  need_gpu qwen36 vk.log "qwen36 COLI_VK_TIER=0"
  echo "OK qwen36 COLI_VK_TIER=0: tokens = CPU, no tier, $(grep -o '[0-9]* matmuls on the GPU' vk.log | tail -1)"

  # qwen38: one fixture, every resident format, with and without prefill batching
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny
  local batch O
  for batch in 0 1; do
    O="OMP_NUM_THREADS=2 SNAP=qwen38_tiny Q38_PREFILL_BATCH=$batch COLI_VK_DENSE=1"   # the trunk's formats, see qwen36 above
    # the default: every fixture matrix is under 1 MiB, so the trunk stays BF16 (fmt 11)
    vk_gate qwen38 'bf16 [1-9]' "qwen38 bf16 batch=$batch" $O -- 1 8 qwen38_tiny/ref.json
    # Q38_TRUNK_MIN_KB=0: the int8 trunk (fmt 1); what stays outside it is BF16
    vk_gate qwen38 'int8 [1-9].*bf16 [1-9]' "qwen38 int8 batch=$batch" $O Q38_TRUNK_MIN_KB=0 -- 1 8 qwen38_tiny/ref.json
    # Q38_NATIVE_BF16=0 expands the rows to f32 at load (fmt 10)
    vk_gate qwen38 'f32 [1-9]' "qwen38 f32 batch=$batch" $O Q38_TRUNK_CPU_INT8=0 Q38_NATIVE_BF16=0 -- 1 8 qwen38_tiny/ref.json
  done

  # The routed-expert tier on qwen38's three expert forms: BF16 (fmt 11), the release's
  # FP8 with 128x128 block scales (fmt 12, gs 128), the experts-int4g64 sidecar's planar
  # int4 (fmt 4); decode one row at a time and prefill batched; at cap=1 with the trunk
  # where the default puts it (the CPU here), at cap=4 with the trunk on the device;
  # with the MTP head drafting (its verify rows take the device's per-row route; the
  # batched one with the trunk on the device); the tier alone; and a budget of two
  # experts, which must evict.
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_fp8 --fp8-experts
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4 --fp8-experts --int4-experts --expert-gain 3
  local fx ref
  for fx in qwen38_tiny qwen38_tiny_fp8 qwen38_tiny_int4; do
    ref=$fx/ref.json; [ $fx = qwen38_tiny_int4 ] && ref=$fx/ref_int4.json
    for batch in 0 1; do for cap in 1 4; do
      D=; [ $cap = 4 ] && D=COLI_VK_DENSE=1
      tier_gate qwen38 "qwen38 tier $fx batch=$batch cap=$cap" OMP_NUM_THREADS=2 $D Q38_PREFILL_BATCH=$batch SNAP=$fx -- $cap 8 $ref
    done; done
  done
  tier_gate qwen38 "qwen38 tier alone (COLI_VK_DENSE=0)" OMP_NUM_THREADS=2 COLI_VK_DENSE=0 SNAP=qwen38_tiny_int4 -- 4 8 qwen38_tiny_int4/ref_int4.json
  EVICT=1 tier_gate qwen38 "qwen38 tier, a budget of two experts" OMP_NUM_THREADS=2 Q38_PREFILL_BATCH=0 COLI_VK_TIER_GB=0.0000065 SNAP=qwen38_tiny_fp8 -- 1 8 qwen38_tiny_fp8/ref.json
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_fp8_mtp --fp8-experts --mtp
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4_mtp --fp8-experts --int4-experts --expert-gain 3 --mtp
  for fx in qwen38_tiny_fp8_mtp qwen38_tiny_int4_mtp; do
    for batch in 0 1; do
      D=; [ $batch = 1 ] && D=COLI_VK_DENSE=1
      tier_gate qwen38 "qwen38 tier MTP $fx batch=$batch" OMP_NUM_THREADS=2 $D Q38_MTP=1 Q38_PREFILL_BATCH=$batch SNAP=$fx -- 2 8 $fx/ref.json
    done
  done
}

# The routed-expert tier under ASan and UBSan: a sanitized VK=1 build of both qwen
# engines, the tier's configurations on Lavapipe (formats, eviction, PILOT's worker
# against the tier, MTP, the tier alone, the trunk on the device or on the CPU). Memory safety is the gate, not the tokens
# (a sanitized build vectorizes differently); each run must still put experts on the
# device, or the tier was never exercised.
family_qwen_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen36 qwen38 VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny64 --ref-mode full --inter 64 --emit-ref qwen36_tiny64/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_c --ebits 4 --gs 64
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_g8 --ebits 8 --gs 64
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_fp8 --fp8-experts
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4 --fp8-experts --int4-experts --expert-gain 3
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4_mtp --fp8-experts --int4-experts --expert-gain 3 --mtp
  san() {  # <engine> <tag> <env and argv...>
    local eng=$1 tag=$2; shift 2
    rm -f tier.usage
    env ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2 \
      COLI_USAGE=tier.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    [ "$(tier_count "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: no routed expert ran on the device"; }
    echo "OK $tag: sanitizers clean, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' san.log | tail -1), $(grep -a -o 'dense matrices on the [a-zA-Z]*' san.log | head -1)"
  }
  local cap D   # cap=8 and batch=1 with the trunk on the device, the others where the default puts it
  for cap in 1 8; do
    D=; [ $cap = 8 ] && D=COLI_VK_DENSE=1
    san qwen36 "asan qwen36 int8 PILOT cap=$cap" $D COLI_DENSE_I8=0 PILOT=1 WIDE=2 SNAP=qwen36_tiny_c ./qwen36 $cap 8 qwen36_tiny/ref_full.json
    san qwen36 "asan qwen36 int4-g64 PILOT cap=$cap" $D COLI_DENSE_I8=0 PILOT=1 WIDE=2 SNAP=qwen36_tiny64_c ./qwen36 $cap 4 qwen36_tiny64/ref_full.json
    san qwen36 "asan qwen36 int8 gs64 cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny64_g8 ./qwen36 $cap 8 qwen36_tiny64/ref_full.json
  done
  san qwen36 "asan qwen36 eviction" COLI_VK_TIER_GB=0.00002 COLI_DENSE_I8=0 PILOT=1 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_full.json
  san qwen36 "asan qwen36 tier alone" COLI_VK_DENSE=0 SNAP=qwen36_tiny64_c ./qwen36 8 4 qwen36_tiny64/ref_full.json
  local b
  for b in 0 1; do
    D=; [ $b = 1 ] && D=COLI_VK_DENSE=1
    san qwen38 "asan qwen38 fp8 batch=$b" $D Q38_PREFILL_BATCH=$b SNAP=qwen38_tiny_fp8 ./qwen38 1 8 qwen38_tiny_fp8/ref.json
    san qwen38 "asan qwen38 int4 batch=$b" $D Q38_PREFILL_BATCH=$b SNAP=qwen38_tiny_int4 ./qwen38 1 8 qwen38_tiny_int4/ref_int4.json
  done
  san qwen38 "asan qwen38 eviction" Q38_PREFILL_BATCH=0 COLI_VK_TIER_GB=0.0000065 SNAP=qwen38_tiny_fp8 ./qwen38 1 8 qwen38_tiny_fp8/ref.json
  san qwen38 "asan qwen38 MTP" COLI_VK_DENSE=1 Q38_MTP=1 Q38_PREFILL_BATCH=0 SNAP=qwen38_tiny_int4_mtp ./qwen38 2 8 qwen38_tiny_int4_mtp/ref.json
  san qwen38 "asan qwen38 tier alone" COLI_VK_DENSE=0 SNAP=qwen38_tiny_int4 ./qwen38 4 8 qwen38_tiny_int4/ref_int4.json
  make clean >/dev/null 2>&1 || true
}

family_inkling_olmoe() {
  make inkling olmoe VK=1
  local cap
  # inkling, f32 fixture (fmt 10): the oracle and the CPU's ids, at three caps
  $PY tools/make_tiny_inkling.py tiny_inkling
  for cap in 1 2 8; do
    SNAP=tiny_inkling ./inkling $cap 0 tiny_inkling/ref_inkling.json > cpu.log 2>&1
    COLI_VULKAN=1 SNAP=tiny_inkling ./inkling $cap 0 tiny_inkling/ref_inkling.json > vk.log 2>&1
    grep -qE 'Matching tokens: ([0-9]+)/\1$' vk.log || { cat vk.log; fail "inkling f32 cap=$cap: oracle"; }
    same_tokens cpu.log vk.log "inkling f32 cap=$cap"
    need_gpu inkling vk.log "inkling f32 cap=$cap"
    echo "OK inkling f32 cap=$cap: $(grep -o 'Matching tokens: [0-9/]*' vk.log), $(vk_count inkling vk.log) matmuls on the GPU"
  done

  # the same weights stored as bf16 (fmt 11). A build with the AVX512-BF16 dot rounds
  # activations to bf16 on the CPU and keeps those matrices there ("0 bf16" in the
  # placement line), so the count is only required when the line places any.
  mkdir -p tiny_inkling_bf16
  cp tiny_inkling/config.json tiny_inkling/generation_config.json tiny_inkling_bf16/
  $PY - <<'EOF'
import json, struct, numpy as np
src, dst = "tiny_inkling/model.safetensors", "tiny_inkling_bf16/model.safetensors"
raw = open(src, "rb").read(); n = struct.unpack("<Q", raw[:8])[0]; hdr = json.loads(raw[8:8 + n])
out, blobs, off = {}, [], 0
for k, v in hdr.items():
    if k == "__metadata__": out[k] = v; continue
    a = np.frombuffer(raw[8 + n + v["data_offsets"][0]: 8 + n + v["data_offsets"][1]], np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    b = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
    out[k] = {"dtype": "BF16", "shape": v["shape"], "data_offsets": [off, off + len(b)]}
    blobs.append(b); off += len(b)
h = json.dumps(out).encode(); h += b" " * (-len(h) % 8)
with open(dst, "wb") as f:
    f.write(struct.pack("<Q", len(h))); f.write(h); [f.write(b) for b in blobs]
EOF
  SNAP=tiny_inkling_bf16 ./inkling 8 0 tiny_inkling/ref_inkling.json > cpu.log 2>&1 || true
  COLI_VULKAN=1 SNAP=tiny_inkling_bf16 ./inkling 8 0 tiny_inkling/ref_inkling.json > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "inkling bf16"
  grep -q ', 0 bf16)' vk.log || need_gpu inkling vk.log "inkling bf16"
  echo "OK inkling bf16: tokens = CPU, $(vk_count inkling vk.log) matmuls on the GPU"

  # the dense-int4g64 container (int8 fmt 1, int4-g64 fmt 4, f32 fmt 10 side by side)
  rm -rf tiny_inkling_q && cp -r tiny_inkling tiny_inkling_q
  $PY - tools/convert_inkling_dense_int4.py tiny_inkling_q <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("conv", sys.argv[1])
conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
conv.MIN_ELEMS = 0
conv.ATTN_BITS = 4
base = conv.classify
conv.classify = lambda n, s, d: "int8" if n.endswith(".mlp.down_proj.weight") else base(n, s, d)
sys.argv = ["convert_inkling_dense_int4.py", "--dir", sys.argv[2]]
conv.main()
EOF
  mkdir -p tiny_inkling_q/dense-int4g64
  mv tiny_inkling_q/dense-int4g64.safetensors tiny_inkling_q/dense-int4g64/dense.safetensors
  SNAP=tiny_inkling_q ./inkling 8 0 tiny_inkling/ref_inkling.json > cpu.log 2>&1 || true
  COLI_VULKAN=1 SNAP=tiny_inkling_q ./inkling 8 0 tiny_inkling/ref_inkling.json > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "inkling int4-g64 container"
  need_gpu inkling vk.log "inkling int4-g64 container"
  echo "OK inkling int4-g64 container: tokens = CPU, $(vk_count inkling vk.log) matmuls on the GPU"

  # olmoe: f32 residents (fmt 10), the oracle and the CPU's ids
  $PY tools/make_olmoe_tiny.py --output olmoe_tiny
  $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c
  SNAP=olmoe_tiny_c ./olmoe 8 8 olmoe_tiny/ref_olmoe.json > cpu.log 2>&1
  COLI_VULKAN=1 SNAP=olmoe_tiny_c ./olmoe 8 8 olmoe_tiny/ref_olmoe.json > vk.log 2>&1
  grep -qE 'Matching tokens: ([0-9]+)/\1$' vk.log || { cat vk.log; fail "olmoe: oracle"; }
  same_tokens cpu.log vk.log "olmoe"
  need_gpu olmoe vk.log "olmoe"
  echo "OK olmoe: $(grep -o 'Matching tokens: [0-9/]*' vk.log), $(vk_count olmoe vk.log) matmuls on the GPU"
}

family_mimo_qwenimage() {
  make mimo qwenimage VK=1
  # mimo: Xiaomi's vendor oracle on the GPU (its engine_env is the f32 dense
  # configuration; its variants include the native FP8/BF16 one and the BF16 vision
  # tower): the trunk on the device with the experts on the CPU (COLI_VK_TIER=0), the
  # routed experts on the shared tier (MXFP4, fmt 7) with the trunk where the default
  # puts it (the CPU on Lavapipe), and both on the device.
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  COLI_VULKAN=1 COLI_VK_TIER=0 $PY tests/mimo_tiny_harness.py --binary ./mimo --fixture ./mimo_tiny
  COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 $PY tests/mimo_tiny_harness.py --binary ./mimo --fixture ./mimo_tiny
  COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DENSE=1 $PY tests/mimo_tiny_harness.py --binary ./mimo --fixture ./mimo_tiny
  # MIMO_DENSE_BITS 0 (native fp8/bf16: fmt 12, 11), 8 (fmt 1) and 32 (fmt 10):
  # the CPU's tokens for every case of ref.json and for the picture, with the experts
  # on the CPU (MIMO_VK_EXPERTS=0, the trunk on the device), on the tier with the trunk
  # on the CPU (the tier must have served some), and on the tier with the trunk on the
  # device (both).
  ids() { $PY -c "import json,sys;r=json.load(open('mimo_tiny/ref.json'));c=r['image'] if sys.argv[1]=='image' else r['cases'][sys.argv[1]];print(' '.join(map(str,c['prompt_ids'])))" "$1"; }
  local grid bits c x extra
  grid=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  for bits in 0 8 32; do
    for c in short window long image; do
      extra=(); [ "$c" = image ] && extra=(--image mimo_tiny/patches.f32 --grid $grid)
      MIMO_DENSE_BITS=$bits COLI_TEMP=0 ./mimo mimo_tiny --ids "$(ids $c)" --ngen 6 "${extra[@]}" > mimo-cpu.txt 2>/dev/null
      for x in MIMO_VK_EXPERTS=0 COLI_VK_DENSE=0 COLI_VK_DENSE=1; do
        env COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 $x MIMO_DENSE_BITS=$bits COLI_TEMP=0 \
          ./mimo mimo_tiny --ids "$(ids $c)" --ngen 6 "${extra[@]}" > mimo-vk.txt 2> mimo-vk.err
        cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-vk.err; fail "mimo bits=$bits $c $x differs from the CPU"; }
        [ $x = COLI_VK_DENSE=0 ] || need_gpu mimo mimo-vk.err "mimo bits=$bits $c $x"
        if [ $x = MIMO_VK_EXPERTS=0 ]; then
          ! grep -q '^\[VK\] tier' mimo-vk.err || { cat mimo-vk.err; fail "mimo bits=$bits $c: MIMO_VK_EXPERTS=0 started the tier"; }
        else
          [ "$(tier_count mimo mimo-vk.err)" -gt 0 ] || { cat mimo-vk.err; fail "mimo bits=$bits $c $x: no routed expert ran on the device"; }
        fi
      done
    done
    echo "OK mimo MIMO_DENSE_BITS=$bits: the CPU's tokens, text and image, experts on the CPU and on the tier, trunk on the device and on the CPU"
  done
  # MIMO_VK_EXPERTS=N sizes the tier at N experts: at 2 it must evict as the routing
  # moves, and still give the CPU's tokens
  MIMO_DENSE_BITS=32 COLI_TEMP=0 ./mimo mimo_tiny --ids "$(ids long)" --ngen 6 > mimo-cpu.txt 2>/dev/null
  COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 MIMO_VK_EXPERTS=2 MIMO_DENSE_BITS=32 COLI_TEMP=0 \
    ./mimo mimo_tiny --ids "$(ids long)" --ngen 6 > mimo-vk.txt 2> mimo-vk.err
  cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-vk.err; fail "mimo MIMO_VK_EXPERTS=2 differs from the CPU"; }
  grep -q 'budget [0-9.]* KiB = 2 experts' mimo-vk.err || { grep '\[VK\]' mimo-vk.err; fail "mimo MIMO_VK_EXPERTS=2: not a budget of two experts"; }
  [ "$(tier_count mimo mimo-vk.err)" -gt 0 ] && [ "$(tier_evictions mimo mimo-vk.err)" -gt 0 ] || { grep '\[VK\] tier' mimo-vk.err; fail "mimo MIMO_VK_EXPERTS=2: the tier served nothing or never evicted"; }
  grep -q ' failed 0 ' mimo-vk.err || { grep '\[VK\] tier' mimo-vk.err; fail "mimo MIMO_VK_EXPERTS=2: an upload failed"; }
  echo "OK mimo, a budget of two experts: tokens = CPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' mimo-vk.err | tail -1), $(grep -a -o 'evictions [0-9]*' mimo-vk.err | tail -1)"

  # qwenimage: at 8, 16 and 32 bits (fmt 1, 11, 10) every oracle stage of the
  # Lavapipe run against the CPU run with the same bits and f32 activations. int8 is
  # outside the oracle's own tolerance on both sides, so the reports are compared
  # with each other, not with the reference; at 16 bits the Vulkan run must also pass.
  $PY tools/make_qwenimage_tiny.py qwenimage_tiny
  for bits in 8 16 32; do
    COLI_IMG_BITS=$bits COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --ref qwenimage_tiny/ref > qi-cpu.log 2>&1 || true
    local rc=0
    COLI_VULKAN=1 COLI_IMG_BITS=$bits COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --ref qwenimage_tiny/ref > qi-vk.log 2>&1 || rc=$?
    [ $bits != 16 ] || [ $rc = 0 ] || { cat qi-vk.log; fail "qwenimage bf16 oracle on Vulkan"; }
    need_gpu qwenimage qi-vk.log "qwenimage oracle bits=$bits"
    BITS=$bits $PY - <<'PY'
import os, re
def stages(p):
    return {k.strip(): float(r) for k, r in
            re.findall(r"\[oracle\] (.+?)\s+n=\d+\s+max\|err\| \S+\s+rel (\S+)", open(p).read())}
c, v = stages("qi-cpu.log"), stages("qi-vk.log")
assert len(c) >= 10 and c.keys() == v.keys(), (c, v)
for k in c:   # relative errors against the reference, CPU vs GPU: equal up to summation order
    assert abs(c[k] - v[k]) <= 0.25 * c[k] + 2e-7, (k, c[k], v[k])
print(f"OK qwenimage oracle: {len(c)} stages of the Lavapipe run match the CPU run at {os.environ['BITS']} bits")
PY
  done
  # a generated picture at the default int8 weights, Lavapipe against the CPU
  COLI_IMG_BITS=8 COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --prompt "a red fox in the snow" \
    --width 256 --height 256 --steps 2 --seed 1 --out qi-cpu.png
  COLI_VULKAN=1 COLI_IMG_BITS=8 COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --prompt "a red fox in the snow" \
    --width 256 --height 256 --steps 2 --seed 1 --out qi-vk.png 2> qi-vk-gen.err
  need_gpu qwenimage qi-vk-gen.err "qwenimage picture"
  $PY - <<'PY'
import sys; sys.path.insert(0, ".")
import image_engine as e
_, _, _, a = e.decode_png(open("qi-cpu.png", "rb").read())
_, _, _, b = e.decode_png(open("qi-vk.png", "rb").read())
d = [abs(x - y) for x, y in zip(a, b)]
assert len(a) == len(b) and max(d) <= 2 and sum(t > 0 for t in d) <= len(d) // 1000, (max(d), sum(t > 0 for t in d))
print(f"OK qwenimage picture: {sum(t > 0 for t in d)} of {len(d)} bytes differ from the CPU's, max {max(d)}")
PY
  # the serve protocol with the device on
  COLI_VULKAN=1 QWENIMAGE_TINY=qwenimage_tiny $PY -m unittest tests.test_qwenimage_engine_serve
}

family_deepseek() {
  make deepseek_v41 VK=1
  # deepseek_v41: fp8 dense in 32x32 ue8m0 tiles (fmt 12, gs 32, the tile scale
  # repeated over its rows) and bf16 (fmt 11). The engine exits non-zero on any
  # token mismatch with the reference; the CPU run must print the same stream.
  $PY tools/make_dsv41_tiny.py --out dsv41_tiny --emit-ref dsv41_tiny/ref.json
  $PY tools/make_dsv41_tiny.py --out dsv41_long --emit-ref dsv41_long/ref.json --prompt-len 40 --max-new 6
  v41() {  # <tag> <env and argv...>
    local tag=$1; shift
    env "$@" > v41-cpu.txt 2> v41-cpu.err || { cat v41-cpu.err; fail "deepseek_v41 $tag: CPU run"; }
    env COLI_VULKAN=1 "$@" > v41-vk.txt 2> v41-vk.err || { cat v41-vk.err; fail "deepseek_v41 $tag: Vulkan run misses the oracle"; }
    cmp -s v41-cpu.txt v41-vk.txt || { diff v41-cpu.txt v41-vk.txt | head; fail "deepseek_v41 $tag: Vulkan output differs from the CPU"; }
    need_gpu deepseek_v41 v41-vk.err "deepseek_v41 $tag"
    echo "OK deepseek_v41 $tag: output = CPU, $(vk_count deepseek_v41 v41-vk.err) matmuls on the GPU"
  }
  local cap force
  for cap in 1 2 8; do v41 "cap=$cap" SNAP=dsv41_tiny ./deepseek_v41 $cap dsv41_tiny/ref.json; done
  for cap in 2 8; do v41 "40-token prompt cap=$cap" SNAP=dsv41_long ./deepseek_v41 $cap dsv41_long/ref.json; done
  for force in 1 2 3 4 5; do
    v41 "DSpark spec=$force" SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=$force ./deepseek_v41 8 dsv41_tiny/ref.json
  done

  # deepseek_v4: fp8 128x128 blocks (fmt 12, gs 128) and the bf16 router, compressors
  # and head (fmt 11). The GPU gets the activations after the CPU's own E4M3 rounding,
  # so the two runs do the same arithmetic. The tiny check builds the VK=1 binary and
  # keeps passing with the device open; its --oracle path reloads the dense weights
  # every forward and so stays on the CPU, which is why the device is checked on the
  # session path below: ids and teacher-forced predictions equal to the CPU's and to
  # the reference's greedy stream.
  COLI_VULKAN=1 make deepseek-v4-tiny-check VK=1
  local prompt
  prompt=$($PY -c 'import json; c=json.load(open("deepseek_v4_tiny/ref.json"))["cases"]["long"]; print("".join("<t%03d>" % t for t in c["prompt_ids"]))')
  ./deepseek_v4 ./deepseek_v4_tiny "$prompt" --raw-prompt --max-tokens 4 --record-oracle v4-cpu.json > /dev/null
  COLI_VULKAN=1 ./deepseek_v4 ./deepseek_v4_tiny "$prompt" --raw-prompt --max-tokens 4 --record-oracle v4-vk.json > /dev/null 2> v4-vk.err
  $PY - <<'PY' || fail "deepseek_v4: the Vulkan session differs from the CPU's"
import json, sys
a, b = json.load(open("v4-cpu.json")), json.load(open("v4-vk.json"))
ref = json.load(open("deepseek_v4_tiny/ref.json"))["cases"]["long"]["greedy_full_ids"]
ok = a["full_ids"] == b["full_ids"] == ref and a["tf_pred"] == b["tf_pred"]
print("OK deepseek_v4 session: ids = CPU = reference" if ok else ("CPU", a, "VK", b, "ref", ref))
sys.exit(0 if ok else 1)
PY
  need_gpu deepseek_v4 v4-vk.err "deepseek_v4 session"
  echo "OK deepseek_v4: $(vk_count deepseek_v4 v4-vk.err) matmuls on the GPU"
}

# Kimi K3: the routed experts on the shared tier (MXFP4 with ue8m0 scales, fmt 7,
# SiTU-GLU in the latent space), against Moonshot's vendor oracle and the CPU run.
# K3_IDOT=0 everywhere: the CPU's default int8-activation expert kernel is an
# approximation the device does not make (the oracle's engine_env sets it too).
family_kimi() {
  make kimi_k3 VK=1
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  # Moonshot's oracle (greedy, teacher forcing at every position, determinism, the
  # bite) with the tier on, the shared experts where the default puts them (the CPU
  # on Lavapipe) and on the device
  COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 \
    $PY tests/test_kimi_k3_tiny.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
  COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DENSE=1 \
    $PY tests/test_kimi_k3_tiny.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
  k3ids() { $PY -c "import json,sys;print(' '.join(map(str,json.load(open('kimi_k3_tiny/ref.json'))['cases'][sys.argv[1]]['prompt_ids'])))" "$1"; }
  # k3_gate <tag> <env...>: the tokens of the CPU run, the tier served some experts,
  # the shared experts where they were asked to be (dense_where); EVICT=1: it evicted
  k3_gate() {
    local tag=$1 c; shift
    for c in short chunk long; do
      rm -f k3.usage
      env "$@" COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 ./kimi_k3 kimi_k3_tiny --ids "$(k3ids $c)" --ngen 8 2>/dev/null | sed 's/ *TUNE.*//' > cpu.tok
      env "$@" COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 \
        ./kimi_k3 kimi_k3_tiny --ids "$(k3ids $c)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
      { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok vk.log; fail "$tag $c: Vulkan tokens differ from the CPU"; }
      [ "$(tier_count kimi_k3 vk.log)" -gt 0 ] || { cat vk.log; fail "$tag $c: no routed expert ran on the device"; }
      if [ "${EVICT:-0}" = 1 ]; then
        [ "$(tier_evictions kimi_k3 vk.log)" -gt 0 ] || { grep '\[VK\] tier' vk.log; fail "$tag $c: the budget forced no eviction"; }
        grep -q ' failed 0 ' vk.log || { grep '\[VK\] tier' vk.log; fail "$tag $c: an upload failed"; }
      fi
      local where; where=$(dense_where kimi_k3 vk.log "$tag $c" "$@") || { echo "$where"; exit 1; }
      echo "OK $tag $c: tokens = CPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.log | tail -1), $(grep -a -o 'evictions [0-9]*' vk.log | tail -1), shared experts on the $where"
    done
  }
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0" b
  k3_gate "kimi_k3 tier f32" $O
  k3_gate "kimi_k3 tier f32, shared experts on the device" $O COLI_VK_DENSE=1
  for b in 8 4; do   # the shared experts as int8 rows (fmt 1) and int4-g64 (fmt 4) on the device
    k3_gate "kimi_k3 tier K3_BITS=$b, shared experts on the device" K3_BITS=$b K3_MLA_BITS=$b K3_HEAD_BITS=$b K3_IDOT=0 COLI_TEMP=0 COLI_VK_DENSE=1
  done
  k3_gate "kimi_k3 tier, prefill one token at a time" $O K3_CHUNK=1
  k3_gate "kimi_k3 tier, loads not pipelined" $O K3_PIPE=0
  EVICT=1 k3_gate "kimi_k3 tier, a budget of two experts" $O COLI_VK_TIER_GB=0.000005
  # the old switches: K3_VK=1 opens the device as COLI_VULKAN=1 does and K3_VK_GB caps
  # the tier; K3_VK=0 keeps it closed whatever COLI_VULKAN says
  rm -f k3.usage
  env $O COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 2>/dev/null | sed 's/ *TUNE.*//' > cpu.tok
  env $O COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 K3_VK=1 K3_VK_GB=0.000005 \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
  cmp -s cpu.tok vk.tok || { cat vk.log; fail "kimi_k3 K3_VK=1: tokens differ from the CPU"; }
  grep -q 'K3_VK=1 read as COLI_VULKAN=1' vk.log && grep -q 'budget [0-9.]* KiB = 2 experts' vk.log &&
    [ "$(tier_count kimi_k3 vk.log)" -gt 0 ] || { cat vk.log; fail "kimi_k3 K3_VK=1 K3_VK_GB: not the tier it asked for"; }
  env $O COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 K3_VK=0 COLI_VULKAN=1 \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
  cmp -s cpu.tok vk.tok && ! grep -q '^\[VK\]' vk.log || { cat vk.log; fail "kimi_k3 K3_VK=0: the device opened"; }
  echo "OK kimi_k3 K3_VK=1 / K3_VK_GB / K3_VK=0: the shared tier's switches"
  # a warm start from the history of the run before: the tier starts full, serves
  # every routed expert of the same prompt, and the tokens stay the CPU's
  rm -f k3.usage
  env $O COLI_USAGE=$PWD/k3.usage COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 > /dev/null 2>&1
  env $O COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
  cmp -s cpu.tok vk.tok || { cat vk.log; fail "kimi_k3 warm start: tokens differ from the CPU"; }
  grep -q 'tier kimi_k3: warm start, [1-9]' vk.log || { cat vk.log; fail "kimi_k3: no warm start from the history"; }
  echo "OK kimi_k3 warm start: tokens = CPU, $(grep -a -o 'warm start, [0-9]* experts' vk.log), $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.log | tail -1)"
  # COLI_VK_TIER=0: the shared experts alone on the device, as before the tier
  env $O COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VK_TIER=0 COLI_VULKAN=1 \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
  cmp -s cpu.tok vk.tok || { cat vk.log; fail "kimi_k3 COLI_VK_TIER=0: tokens differ from the CPU"; }
  ! grep -q '^\[VK\] tier' vk.log || { cat vk.log; fail "kimi_k3 COLI_VK_TIER=0: the tier started"; }
  need_gpu kimi_k3 vk.log "kimi_k3 COLI_VK_TIER=0"
  echo "OK kimi_k3 COLI_VK_TIER=0: tokens = CPU, no tier, $(vk_count kimi_k3 vk.log) matmuls on the GPU"
}

# The Kimi K3 and MiMo expert tiers under ASan and UBSan, as qwen-sanitize does for
# the qwen engines: memory safety is the gate, and each run must still put routed
# experts on the device.
family_kimi_mimo_sanitize() {
  make clean >/dev/null 2>&1 || true
  make kimi_k3 mimo VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  ksan() {  # <engine> <tag> <env and argv...>
    local eng=$1 tag=$2; shift 2
    env ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2 \
      COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    [ "$(tier_count "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: no routed expert ran on the device"; }
    echo "OK $tag: sanitizers clean, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' san.log | tail -1), $(grep -a -o 'evictions [0-9]*' san.log | tail -1)"
  }
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0" K3IDS MIDS IMG GRID
  K3IDS=$($PY -c "import json;print(' '.join(map(str,json.load(open('kimi_k3_tiny/ref.json'))['cases']['long']['prompt_ids'])))")
  MIDS=$($PY -c "import json;print(' '.join(map(str,json.load(open('mimo_tiny/ref.json'))['cases']['long']['prompt_ids'])))")
  IMG=$($PY -c "import json;print(' '.join(map(str,json.load(open('mimo_tiny/ref.json'))['image']['prompt_ids'])))")
  GRID=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  rm -f k3.usage
  ksan kimi_k3 "asan kimi_k3 tier" $O COLI_USAGE=$PWD/k3.usage ./kimi_k3 kimi_k3_tiny --ids "$K3IDS" --ngen 8
  ksan kimi_k3 "asan kimi_k3 warm start, shared experts on the device" $O COLI_USAGE=$PWD/k3.usage COLI_VK_DENSE=1 ./kimi_k3 kimi_k3_tiny --ids "$K3IDS" --ngen 8
  ksan kimi_k3 "asan kimi_k3 eviction, int8 shared experts" K3_BITS=8 K3_IDOT=0 COLI_USAGE=$PWD/k3e.usage USAGE_SAVE=0 COLI_VK_TIER_GB=0.000005 COLI_VK_DENSE=1 ./kimi_k3 kimi_k3_tiny --ids "$K3IDS" --ngen 8
  ksan kimi_k3 "asan kimi_k3 prefill one token at a time, no pipeline" $O COLI_USAGE=$PWD/k3e.usage USAGE_SAVE=0 K3_CHUNK=1 K3_PIPE=0 ./kimi_k3 kimi_k3_tiny --ids "$K3IDS" --ngen 8
  ksan mimo "asan mimo tier" MIMO_DENSE_BITS=32 COLI_TEMP=0 ./mimo mimo_tiny --ids "$MIDS" --ngen 6
  ksan mimo "asan mimo eviction (MIMO_VK_EXPERTS=2)" MIMO_DENSE_BITS=32 COLI_TEMP=0 MIMO_VK_EXPERTS=2 ./mimo mimo_tiny --ids "$MIDS" --ngen 6
  ksan mimo "asan mimo picture, native dense on the device" MIMO_DENSE_BITS=0 COLI_TEMP=0 COLI_VK_DENSE=1 ./mimo mimo_tiny --ids "$IMG" --ngen 6 --image mimo_tiny/patches.f32 --grid $GRID
  ksan mimo "asan mimo prefill blocks of 3, cache 4" MIMO_DENSE_BITS=32 COLI_TEMP=0 MIMO_CHUNK=3 MIMO_CAP=4 ./mimo mimo_tiny --ids "$MIDS" --ngen 6
  make clean >/dev/null 2>&1 || true
}

case "${1:-}" in
  shader)         shader_formats ;;
  qwen)           family_qwen ;;
  qwen-sanitize)  family_qwen_sanitize ;;
  inkling-olmoe)  family_inkling_olmoe ;;
  mimo-qwenimage) family_mimo_qwenimage ;;
  deepseek)       family_deepseek ;;
  kimi)           family_kimi ;;
  kimi-mimo-sanitize) family_kimi_mimo_sanitize ;;
  *) echo "usage: $0 shader|qwen|qwen-sanitize|inkling-olmoe|mimo-qwenimage|deepseek|kimi|kimi-mimo-sanitize" >&2; exit 2 ;;
esac
