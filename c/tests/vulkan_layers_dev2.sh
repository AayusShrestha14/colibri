# Sourced by tests/vulkan_engines.sh: the dense chain's layers on two devices (docs/vulkan.md,
# "Layers on two devices"). Lavapipe is opened twice (COLI_VK_DEV2=0: a second logical device
# on the same physical one, the test mode); COLI_VK_CHAIN_LAYERS puts the first layers on
# the primary device, COLI_VK_CHAIN_LAYERS2 the next ones on the second, the CPU the rest.
#
#   layers-dev2            every chain engine against its own CPU run
#   layers-dev2-sanitize   the same paths under ASan and UBSan
#
# Each gate (ld2_gate; ld2_mla for the MLA engines) is chain_gate's or mla_gate's (the
# CPU's tokens, logits within 1e-4 of the largest) with the split forced, and the second
# device's chain must have run with neither device lost.

# ld2_gate <engine> <tag> <n0> <n1> <env...> -- <argv...>
ld2_gate() {
  local eng=$1 tag=$2 n0=$3 n1=$4; shift 4
  chain_gate "$eng" "$tag" 1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 "$@"
  ld2_ran "$eng" "$tag" vk.log
}
# ld2_mla <engine> <tag> <n0> <n1> <env...> -- <argv...>
ld2_mla() {
  local eng=$1 tag=$2 n0=$3 n1=$4; shift 4
  mla_gate "$eng" "$tag" 1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 "$@"
  ld2_ran "$eng" "$tag" vk.log
}
# ld2_ran <engine> <tag> <log>: the second device's chain ran, no device was lost
ld2_ran() {
  local f; f=$(ld2_forwards "$1" "$3")
  [ "$f" -gt 0 ] || { cat "$3"; fail "$2: the second device's chain never ran"; }
  ! grep -qa 'the device is lost\|the device was lost' "$3" || { grep -a 'lost' "$3"; fail "$2: a device was lost"; }
  echo "   $2: the second device's chain ran $f forwards"
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
  case $eng in
    colibri) mla_toks "$eng" cpu.log > cpu.tok; mla_toks "$eng" vk.log > vk.tok
             { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -20 vk.log; fail "$tag: the tokens differ from the CPU"; } ;;
    *) same_tokens cpu.log vk.log "$tag" ;;
  esac
  grep -q "COLI_VK_CHAIN_FAULT2" vk.log || { cat vk.log; fail "$tag: the second device's fault never fired"; }
  if [ "${LD2_EXPECT:-}" = setup ]; then
    grep -q "^\[VK\] $eng dev2 chain: 0 of [0-9]* layers on the device: the chain stays off (layer 0 did not reach the device: the device was lost" vk.log ||
      { cat vk.log; fail "$tag: the second device's layers did not go to the CPU"; }
    echo "OK $tag: tokens = CPU, the second device's layers on the CPU from the start"
    return 0
  fi
  if [ "${LD2_EXPECT:-}" = again ]; then   # the KV cache is the host's: the CPU runs the forward again
    grep -q "^\[VK\] $eng dev2 chain: the device was lost; the CPU runs this forward again" vk.log ||
      { cat vk.log; fail "$tag: the CPU did not take the forward over"; }
    echo "OK $tag: tokens = CPU, the CPU ran the forward again"
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

# mimo: its own gate (mimo_chain_gate: the vendor fixture's cases, tokens and logits), with
# the split forced; the second device's chain must have run in the logits run
ld2_mimo_gate() {   # <tag> <case> <n0> <n1> <env...>
  local tag=$1 c=$2 n0=$3 n1=$4; shift 4
  mimo_chain_gate "$tag" "$c" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 "$@"
  ld2_ran mimo "$tag $c" mimo-vk.err
}
ld2_mimo() {
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  MGRID=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  local c
  for c in $($PY -c "import json;print(' '.join(json.load(open('mimo_tiny/ref.json'))['cases']))"); do
    ld2_mimo_gate "ld2 mimo 3 + 3, the head on the second device" $c 3 3 MIMO_DENSE_BITS=32
    ld2_mimo_gate "ld2 mimo 1 + 2, the CPU the rest and the head" $c 1 2 MIMO_DENSE_BITS=32
  done
  ld2_mimo_gate "ld2 mimo 2 + 4, blocks of 3" image 2 4 MIMO_DENSE_BITS=32 MIMO_CHUNK=3
  ld2_mimo_gate "ld2 mimo 1 + 5, the release's FP8 and BF16" image 1 5 MIMO_DENSE_BITS=0
  ld2_mimo_gate "ld2 mimo experts on both devices too" image 3 3 MIMO_DENSE_BITS=32 MIMO_VK_EXPERTS=2 COLI_VK_TIER_GB=0.00002
  ld2_mimo_gate "ld2 mimo KV split on both" image 3 3 MIMO_DENSE_BITS=32 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2
  CHAINMODE=2 ld2_mimo_gate "ld2 mimo prompts only" image 3 3 MIMO_DENSE_BITS=32
  # the second device lost at its third frame: the host's caches hold whole steps, the CPU
  # runs the rest from where they end
  local x=(--image mimo_tiny/patches.f32 --grid $MGRID) ids
  ids=$(mimo_ids image prompt_ids)
  COLI_TEMP=0 MIMO_DENSE_BITS=32 ./mimo mimo_tiny --ids "$ids" --ngen 6 "${x[@]}" > mimo-cpu.txt 2>/dev/null
  COLI_TEMP=0 MIMO_DENSE_BITS=32 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=3 \
    COLI_VK_CHAIN_LAYERS2=3 COLI_VK_CHAIN_FAULT2=3 ./mimo mimo_tiny --ids "$ids" --ngen 6 "${x[@]}" > mimo-vk.txt 2> mimo-vk.err
  cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-cpu.txt mimo-vk.txt mimo-vk.err; fail "ld2 mimo second device lost: tokens differ from the CPU"; }
  grep -q "COLI_VK_CHAIN_FAULT2" mimo-vk.err && grep -q "the device was lost at position" mimo-vk.err ||
    { cat mimo-vk.err; fail "ld2 mimo second device lost: the loss was not taken over"; }
  echo "OK ld2 mimo second device lost: tokens = CPU, $(grep -o 'the device was lost at position [0-9]*' mimo-vk.err)"
}

ld2_inkling() {
  [ -f tiny_inkling/ref_long.json ] || kv_inkling_fixtures
  local R=tiny_inkling/ref_inkling.json T=SNAP=tiny_inkling b
  ld2_gate inkling "ld2 inkling 3 + 5, lm_head on the second device" 3 5 $T -- 8 0 $R
  ld2_gate inkling "ld2 inkling 2 + 2, the CPU the rest and lm_head" 2 2 $T -- 8 0 $R
  ld2_gate inkling "ld2 inkling 1 + 7, prompt in chunks of 3" 1 7 $T COLI_VK_CHAIN_ROWS=3 -- 8 0 $R
  for b in 4 8; do ld2_gate inkling "ld2 inkling runtime int$b" 3 5 $T -- 2 $b $R; done
  ld2_gate inkling "ld2 inkling experts on both devices too" 3 5 $T COLI_VK_TIER_GB=0.00002 -- 8 0 $R
  CHAINMODE=2 ld2_gate inkling "ld2 inkling prompts only" 3 5 $T -- 8 0 $R
  # the global layer's KV split (the fixture has one global layer, 5): on the second
  # device, then on the first
  ld2_gate inkling "ld2 inkling KV split on the second device" 4 4 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 $T -- 8 0 tiny_inkling/ref_long.json
  [ "$(kv_hostparts "inkling dev2" vk.log)" -gt 0 ] || { cat vk.log; fail "ld2 inkling KV split on the second device: no host part"; }
  ld2_gate inkling "ld2 inkling KV split on the first device" 6 2 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 $T -- 8 0 tiny_inkling/ref_long.json
  [ "$(kv_hostparts inkling vk.log)" -gt 0 ] || { cat vk.log; fail "ld2 inkling KV split on the first device: no host part"; }
  LD2_EXPECT=setup ld2_lost inkling "ld2 inkling second device lost at its setup" 3 5 1 $T -- 8 0 $R
  ld2_lost inkling "ld2 inkling second device lost mid-decode" 3 5 30 $T -- 8 0 $R
  CHAIN_SERVE_EXPECT='inkling dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=5
}

# colibri (GLM-5.2): the head stays on the host. glm_tiny_shx shares indexers (full,
# shared, full, shared, shared): with DSA top-4 the selection of the layer before the
# second device's first one crosses to it.
ld2_colibri() {
  [ -f glm_tiny_serve/tokenizer.json ] || glm_chain_fixtures
  [ -d glm_tiny_shx ] || ptl_glm_shx glm_tiny glm_tiny_shx
  export CAP_RAISE=0
  local G="SNAP=glm_tiny REF=ref_glm.json USAGE_SAVE=0" X="SNAP=glm_tiny_shx REF=ref_glm.json USAGE_SAVE=0" s
  ld2_mla colibri "ld2 colibri 2 + 3, every layer on the devices" 2 3 $G -- 64 16 16
  ld2_mla colibri "ld2 colibri 1 + 2, the CPU the rest" 1 2 $G -- 64 16 16
  ld2_mla colibri "ld2 colibri cap 1" 3 2 $G -- 1 16 16
  ld2_mla colibri "ld2 colibri prefill in chunks of 3" 2 3 $G TF=1 COLI_VK_CHAIN_ROWS=3 -- 64 16 16
  ld2_mla colibri "ld2 colibri 4-bit trunk and experts" 2 3 $G IDOT=0 -- 2 4 4
  ld2_mla colibri "ld2 colibri i4 container" 2 3 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json USAGE_SAVE=0 IDOT=0 -- 1 4 4
  ld2_mla colibri "ld2 colibri DSA top-4" 2 3 $G DSA_TOPK=4 -- 64 16 16
  ld2_mla colibri "ld2 colibri DSA_FORCE" 2 3 $G DSA_FORCE=1 -- 64 16 16
  for s in "1 2" "3 2"; do
    ld2_mla colibri "ld2 colibri DSA top-4, a shared indexer first on the second device, ${s/ / + }" ${s% *} ${s#* } $X DSA_TOPK=4 -- 64 16 16
    ld2_mla colibri "ld2 colibri DSA top-4, a shared indexer first on the second device, ${s/ / + }, prefill in chunks of 5" \
      ${s% *} ${s#* } $X DSA_TOPK=4 TF=1 COLI_VK_CHAIN_ROWS=5 -- 64 16 16
  done
  ld2_mla colibri "ld2 colibri n-gram drafts" 2 3 $G DRAFT=3 -- 64 16 16
  ld2_mla colibri "ld2 colibri MTP depth 2" 2 3 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json USAGE_SAVE=0 DRAFT=2 -- 64 16 16
  ld2_mla colibri "ld2 colibri MTP with DSA top-4" 1 2 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json USAGE_SAVE=0 DSA_TOPK=4 -- 64 16 16
  ld2_mla colibri "ld2 colibri experts on both devices too" 2 3 $G COLI_VK_TIER_GB=0.00002 -- 64 16 16
  CHAINMODE=2 ld2_mla colibri "ld2 colibri prompts only, drafts" 2 3 $G DRAFT=3 -- 64 16 16
  ld2_mla colibri "ld2 colibri KV split on both" 2 3 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 $G DSA_TOPK=4 -- 64 16 16
  [ "$(kv_hostparts colibri vk.log)" -gt 0 ] && [ "$(kv_hostparts "colibri dev2" vk.log)" -gt 0 ] ||
    { cat vk.log; fail "ld2 colibri KV split on both: a device's split ran no host part"; }
  # the second device lost: at its setup, in the prompt (one forward), mid-decode
  LD2_EXPECT=setup ld2_lost colibri "ld2 colibri second device lost at its setup" 2 2 1 $G -- 64 16 16
  LD2_EXPECT=again ld2_lost colibri "ld2 colibri second device lost in the prompt" 2 2 2 $G TF=1 -- 64 16 16
  LD2_EXPECT=again ld2_lost colibri "ld2 colibri second device lost mid-decode" 2 2 10 $G -- 64 16 16
  LD2_EXPECT=again ld2_lost colibri "ld2 colibri second device lost with MTP" 2 2 10 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json \
    USAGE_SAVE=0 DRAFT=2 -- 64 16 16
  # serve sessions: pins, the prompt cache, the prefill read-out, two KV slots
  CHAIN_SERVE_EXPECT='colibri dev2 chain: [1-9][0-9]* forwards' CHAIN_SERVE_DIALECT=colibri \
    $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=3
  CHAIN_SERVE_EXPECT='colibri dev2 chain: [1-9][0-9]* forwards' CHAIN_SERVE_DIALECT=colibri \
    $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 DSA_TOPK=4 DRAFT=3 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=2
  CHAIN_SERVE_EXPECT='colibri dev2 chain: [1-9][0-9]* forwards' CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=colibri \
    $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 KV_SLOTS=2 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2
}

family_layers_dev2() {
  export OMP_NUM_THREADS=2
  make qwen36 qwen38 olmoe mimo inkling colibri tests/test_vk_chain VK=1
  COLI_VK_DEV2=0 ./tests/test_vk_chain shaders/qmatmul.spv | tee vk_chain.log
  tail -1 vk_chain.log | grep -qx PASS || fail "the chain's ops on two devices"
  ld2_qwen36
  ld2_qwen38
  ld2_olmoe
  ld2_mimo
  ld2_inkling
  ld2_colibri
  unset OMP_NUM_THREADS
}
family_layers_dev2_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen36 qwen38 olmoe mimo inkling VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
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
  [ -f tiny_inkling/ref_long.json ] || kv_inkling_fixtures
  kv_san inkling "asan ld2 inkling KV split on the second device" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=4 COLI_VK_CHAIN_LAYERS2=4 \
    COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=tiny_inkling ./inkling 8 0 tiny_inkling/ref_long.json
  [ "$(ld2_forwards inkling san.log)" -gt 0 ] || { cat san.log; fail "asan ld2 inkling: the second device's chain never ran"; }
  make clean >/dev/null 2>&1 || true
}
