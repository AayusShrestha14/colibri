# Sourced by tests/vulkan_engines.sh: the dense chain's layers on two devices (docs/vulkan.md,
# "Layers on two devices"). Lavapipe is opened twice (COLI_VK_DEV2=0: a second logical device
# on the same physical one, the test mode); COLI_VK_CHAIN_LAYERS puts the first layers on
# the primary device, COLI_VK_CHAIN_LAYERS2 the next ones on the second, the CPU the rest.
#
#   layers-dev2            every chain engine against its own CPU run
#   layers-dev2-sanitize   the same paths under ASan and UBSan
#
# Each gate (ld2_gate) is chain_gate's (the CPU's tokens, logits within 1e-4 of the
# largest) with the split forced, and the second device's chain must have run.

# ld2_gate <engine> <tag> <n0> <n1> <env...> -- <argv...>
ld2_gate() {
  local eng=$1 tag=$2 n0=$3 n1=$4; shift 4
  chain_gate "$eng" "$tag" 1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 "$@"
  local f; f=$(ld2_forwards "$eng" vk.log)
  [ "$f" -gt 0 ] || { cat vk.log; fail "$tag: the second device's chain never ran"; }
  echo "   $tag: the second device's chain ran $f forwards"
}
# ld2_forwards <engine> <log>: N from "[VK] <engine> dev2 chain: N forwards"
ld2_forwards() {
  local n; n=$(sed -n "s/^\[VK\] $1 dev2 chain: \([0-9]*\) forwards.*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}
# ld2_lost <engine> <tag> <n0> <n1> <frame> <env...> -- <argv...>: the second device lost at
# its given frame (COLI_VK_CHAIN_FAULT2): the CPU's tokens; at its first frame (the setup's
# parameter upload) LD2_EXPECT=setup: its layers stay on the CPU, nothing to rebuild; past
# it, the state rebuilt on the CPU (LD2_EXPECT=redo, an engine without recurrent state:
# the CPU redoes the step)
ld2_lost() {
  local eng=$1 tag=$2 n0=$3 n1=$4 k=$5; shift 5
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage
  env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_CHAIN_FAULT2=$k \
    COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 ./"$eng" "$@" > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "$tag"
  grep -q "COLI_VK_CHAIN_FAULT2" vk.log || { cat vk.log; fail "$tag: the second device's fault never fired"; }
  if [ "${LD2_EXPECT:-}" = setup ]; then
    grep -q "^\[VK\] $eng dev2 chain: 0 of [0-9]* layers on the device: the chain stays off (layer 0 did not reach the device: the device was lost" vk.log ||
      { cat vk.log; fail "$tag: the second device's layers did not go to the CPU"; }
    echo "OK $tag: tokens = CPU, the second device's layers on the CPU from the start"
    return 0
  fi
  if [ "${LD2_EXPECT:-}" = redo ]; then
    grep -q "the CPU redoes the step" vk.log || { cat vk.log; fail "$tag: the CPU did not redo the step"; }
    echo "OK $tag: tokens = CPU, the CPU redid the step"
    return 0
  fi
  grep -q "rebuilding the state of [1-9]" vk.log || { cat vk.log; fail "$tag: no state was rebuilt"; }
  echo "OK $tag: tokens = CPU, $(grep -o 'rebuilding the state of [0-9]* positions' vk.log | head -1)"
}

ld2_qwen36() {
  [ -d qwen36_tiny_c ] || { $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
                            $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8; }
  [ -d qwen36_kv_c ] || kv_qwen36_fixtures
  local R=qwen36_tiny/ref_full.json T="SNAP=qwen36_tiny_c COLI_DENSE_I8=0"
  ld2_gate qwen36 "ld2 qwen36 3 + 5, the head on the second device" 3 5 $T -- 8 8 $R
  ld2_gate qwen36 "ld2 qwen36 2 + 3, the CPU the rest and the head" 2 3 $T -- 8 8 $R
  ld2_gate qwen36 "ld2 qwen36 1 + 7, prompt in chunks of 3" 1 7 $T COLI_VK_CHAIN_ROWS=3 -- 8 8 $R
  ld2_gate qwen36 "ld2 qwen36 cap 1" 4 4 $T -- 1 8 $R
  CPUENV=COLI_DENSE_IDOT=0 ld2_gate qwen36 "ld2 qwen36 int8 trunk" 4 4 SNAP=qwen36_tiny_c -- 8 8 $R
  ld2_gate qwen36 "ld2 qwen36 experts on both devices too" 3 5 $T COLI_VK_TIER_GB=0.00002 -- 8 8 $R
  CHAINMODE=2 ld2_gate qwen36 "ld2 qwen36 prompts only" 3 5 $T -- 8 8 $R
  ld2_gate qwen36 "ld2 qwen36 dense model" 2 4 SNAP=qwen38_27b_kv_c COLI_DENSE_I8=0 -- 8 8 qwen38_27b_kv/ref_full.json
  # the KV cache split on both devices' attention layers
  ld2_gate qwen36 "ld2 qwen36 KV split on both" 4 4 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=qwen36_kv_c COLI_DENSE_I8=0 \
    -- 8 8 qwen36_kv/ref_full.json
  [ "$(kv_hostparts qwen36 vk.log)" -gt 0 ] && [ "$(kv_hostparts "qwen36 dev2" vk.log)" -gt 0 ] ||
    { cat vk.log; fail "ld2 qwen36 KV split on both: a device's split ran no host part"; }
  # prompt-lookup drafts verified across both devices (rollbacks of both chains' copies)
  local f
  for f in accept mixed cycle row1 row3; do
    ld2_gate qwen36 "ld2 qwen36 lookup $f" 3 5 COLI_DENSE_I8=0 COLI_LOOKUP=1 COLI_SPEC_GATE=0 COLI_LOOKUP_FORCE=$f \
      SNAP=qwen36_kv_c -- 8 8 qwen36_kv/ref_full.json
  done
  # the second device lost: at its first frame, and mid-decode
  LD2_EXPECT=setup ld2_lost qwen36 "ld2 qwen36 second device lost at its setup" 3 5 1 $T -- 8 8 $R
  ld2_lost qwen36 "ld2 qwen36 second device lost mid-decode" 3 5 40 SNAP=qwen36_kv_c COLI_DENSE_I8=0 -- 8 8 qwen36_kv/ref_full.json
  # the prefix-reuse contract with the state on both devices, and serve sessions (pins,
  # prompt-cache extensions, a divergent prompt, logprobs)
  QWEN36_TINY=$PWD/qwen36_tiny_c COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=5 \
    COLI_USAGE=$PWD/chain.usage $PY tests/test_qwen36_prefix_serve.py
  CHAIN_SERVE_EXPECT='qwen36 dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./qwen36 qwen36_tiny_c COLI_DENSE_I8=0 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=5
}

ld2_qwen38() {
  [ -d qwen38_tiny ] || $PY tools/make_qwen38_tiny.py --out qwen38_tiny
  [ -d qwen38_tiny_fp8 ] || $PY tools/make_qwen38_tiny.py --out qwen38_tiny_fp8 --fp8-experts
  [ -d qwen38_kv_mtp ] || kv_qwen38_fixtures
  local R=qwen38_tiny/ref.json T="OMP_NUM_THREADS=2 SNAP=qwen38_tiny" f b
  for b in 0 1; do
    ld2_gate qwen38 "ld2 qwen38 2 + 2, the head on the second device, batch=$b" 2 2 $T Q38_PREFILL_BATCH=$b -- 4 8 $R
  done
  ld2_gate qwen38 "ld2 qwen38 1 + 2, the CPU the rest and the head" 1 2 $T -- 4 8 $R
  ld2_gate qwen38 "ld2 qwen38 1 + 3, prompt in chunks of 3" 1 3 $T COLI_VK_CHAIN_ROWS=3 -- 4 8 $R
  ld2_gate qwen38 "ld2 qwen38 bf16 rows" 2 2 $T Q38_NATIVE_BF16=1 -- 4 8 $R
  ld2_gate qwen38 "ld2 qwen38 fp8 experts" 2 2 OMP_NUM_THREADS=2 SNAP=qwen38_tiny_fp8 -- 2 8 qwen38_tiny_fp8/ref.json
  ld2_gate qwen38 "ld2 qwen38 experts on both devices too" 2 2 $T COLI_VK_TIER_GB=0.00002 -- 4 8 $R
  CHAINMODE=2 ld2_gate qwen38 "ld2 qwen38 prompts only" 2 2 $T -- 4 8 $R
  # MTP drafts verified across both devices (both chains' copies rolled back)
  for f in "" accept reject mixed; do
    ld2_gate qwen38 "ld2 qwen38 MTP ${f:-drafting}" 2 2 OMP_NUM_THREADS=2 Q38_MTP=1 Q38_MTP_FORCE=$f SNAP=qwen38_tiny_mtp -- 2 8 qwen38_tiny_mtp/ref.json
  done
  # the KV split on both devices' QSA layers (layers 1 and 3), with MTP
  ld2_gate qwen38 "ld2 qwen38 KV split on both" 2 2 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=qwen38_kv -- 4 8 qwen38_kv/ref.json
  [ "$(kv_hostparts qwen38 vk.log)" -gt 0 ] && [ "$(kv_hostparts "qwen38 dev2" vk.log)" -gt 0 ] ||
    { cat vk.log; fail "ld2 qwen38 KV split on both: a device's split ran no host part"; }
  ld2_gate qwen38 "ld2 qwen38 KV split on both, MTP mixed" 2 2 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 Q38_MTP=1 \
    Q38_MTP_FORCE=mixed SNAP=qwen38_kv_mtp -- 4 8 qwen38_kv_mtp/ref.json
  # the second device lost: at its setup, mid-decode, and in a verify
  LD2_EXPECT=setup ld2_lost qwen38 "ld2 qwen38 second device lost at its setup" 2 2 1 $T -- 4 8 $R
  ld2_lost qwen38 "ld2 qwen38 second device lost mid-decode" 2 2 20 $T -- 4 8 $R
  ld2_lost qwen38 "ld2 qwen38 second device lost in a verify" 2 2 15 OMP_NUM_THREADS=2 Q38_MTP=1 Q38_MTP_FORCE=mixed \
    SNAP=qwen38_tiny_mtp -- 2 8 qwen38_tiny_mtp/ref.json
  # serve sessions with MTP and without
  CHAIN_SERVE_EXPECT='qwen38 dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2
  CHAIN_SERVE_EXPECT='qwen38 dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=2
}

ld2_olmoe() {
  [ -d olmoe_tiny_c ] || { $PY tools/make_olmoe_tiny.py --output olmoe_tiny; $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c; }
  [ -f olmoe_tiny/ref_olmoe_long.json ] || kv_olmoe_fixtures
  local OR=olmoe_tiny/ref_olmoe.json T=SNAP=olmoe_tiny_c cap
  ld2_gate olmoe "ld2 olmoe 2 + 2, the head on the second device" 2 2 $T -- 8 8 $OR
  ld2_gate olmoe "ld2 olmoe 1 + 2, the CPU the rest and the head" 1 2 $T -- 8 8 $OR
  ld2_gate olmoe "ld2 olmoe 1 + 3, prompt in chunks of 3" 1 3 $T COLI_VK_CHAIN_ROWS=3 -- 8 8 $OR
  for cap in 1 2; do ld2_gate olmoe "ld2 olmoe PILOT cap=$cap" 2 2 PILOT=1 WIDE=2 $T -- $cap 8 $OR; done
  ld2_gate olmoe "ld2 olmoe 4-bit experts" 2 2 $T -- 8 4 $OR
  ld2_gate olmoe "ld2 olmoe experts on both devices too" 2 2 $T COLI_VK_TIER_GB=0.000025 -- 8 8 $OR
  CHAINMODE=2 ld2_gate olmoe "ld2 olmoe prompts only" 2 2 $T -- 8 8 $OR
  ld2_gate olmoe "ld2 olmoe KV split on both" 2 2 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 $T -- 8 8 olmoe_tiny/ref_olmoe_long.json
  [ "$(kv_hostparts olmoe vk.log)" -gt 0 ] && [ "$(kv_hostparts "olmoe dev2" vk.log)" -gt 0 ] ||
    { cat vk.log; fail "ld2 olmoe KV split on both: a device's split ran no host part"; }
  LD2_EXPECT=setup ld2_lost olmoe "ld2 olmoe second device lost at its setup" 2 2 1 $T -- 8 8 $OR
  LD2_EXPECT=redo ld2_lost olmoe "ld2 olmoe second device lost mid-decode" 2 2 8 $T -- 8 8 $OR
  CHAIN_SERVE_EXPECT='olmoe dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2
  CHAIN_SERVE_EXPECT='olmoe dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c PILOT=1 WIDE=2 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=2
}

family_layers_dev2() {
  export OMP_NUM_THREADS=2
  make qwen36 qwen38 olmoe tests/test_vk_chain VK=1
  COLI_VK_DEV2=0 ./tests/test_vk_chain shaders/qmatmul.spv | tee vk_chain.log
  tail -1 vk_chain.log | grep -qx PASS || fail "the chain's ops on two devices"
  ld2_qwen36
  ld2_qwen38
  ld2_olmoe
  unset OMP_NUM_THREADS
}
family_layers_dev2_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen36 qwen38 olmoe VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  [ -d qwen36_kv_c ] || kv_qwen36_fixtures
  kv_san qwen36 "asan ld2 qwen36 KV split on both" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=4 COLI_VK_CHAIN_LAYERS2=4 \
    COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 COLI_DENSE_I8=0 SNAP=qwen36_kv_c ./qwen36 8 8 qwen36_kv/ref_full.json
  [ "$(ld2_forwards qwen36 san.log)" -gt 0 ] || { cat san.log; fail "asan ld2 qwen36: the second device's chain never ran"; }
  [ -d qwen38_kv_mtp ] || kv_qwen38_fixtures
  kv_san qwen38 "asan ld2 qwen38 KV split on both, MTP mixed" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2 \
    COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 Q38_MTP=1 Q38_MTP_FORCE=mixed SNAP=qwen38_kv_mtp ./qwen38 4 8 qwen38_kv_mtp/ref.json
  [ "$(ld2_forwards qwen38 san.log)" -gt 0 ] || { cat san.log; fail "asan ld2 qwen38: the second device's chain never ran"; }
  [ -f olmoe_tiny/ref_olmoe_long.json ] || kv_olmoe_fixtures
  kv_san olmoe "asan ld2 olmoe KV split on both, PILOT" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2 \
    COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 PILOT=1 WIDE=2 SNAP=olmoe_tiny_c ./olmoe 2 8 olmoe_tiny/ref_olmoe_long.json
  [ "$(ld2_forwards olmoe san.log)" -gt 0 ] || { cat san.log; fail "asan ld2 olmoe: the second device's chain never ran"; }
  make clean >/dev/null 2>&1 || true
}
