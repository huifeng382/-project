#!/bin/bash
# 17.6.0 validation round 2. Round 1 was VOID: XYCE_TIMEOUT_S=1 is LONGER than a real
# single sim at level0 (tens of ms), so it never fired and only served as a negative control.
# t = single-sim timeout inside the REAL pipeline (XYCE_BIN -> ~/slowsim.sh sleep 5), plus the
#     consecutive-5 abort. TL_MAX_STEPS=2 + TL_MAX_WALL_S=300 are belts: if the wrapper were
#     somehow ignored, this leg still ends in minutes instead of hours.
# n = normal path, 40 steps, no knob fires.
# w = circuit wall clock fires mid-run (own CACHE dir so it stays cold and slow).
# ASCII only. Run on server in ~/NetlistOpt. Usage: bash ~/bfval2.sh
set -u
cd "$HOME/NetlistOpt" || exit 1
export OMP_NUM_THREADS=6
V="$PWD/temp_sim_test/bfval2"
K="$PWD/temp_sim_cache_bfval2"
mkdir -p "$V" "$K"
B="GNN_SHADOW=1 TL_BESTFIRST=1 TL_MAX_STEPS=40"
T="cargo test --release --test tl_opt_gnn_batch -- --nocapture --ignored"
echo "B=$B"
echo "T=$T"
mk() {
  tag="$1"; shift
  echo "=== RUN $tag ==="
  echo "env: $*"
  s=$(date +%s)
  env $B "$@" SHADOW_OUT_BASE="$V/$tag" XYCE_CACHE_DIR="$K/$tag" $T > "$HOME/b2_$tag.log" 2>&1
  rc=$?
  echo "rc=$rc wall_s=$(( $(date +%s) - s ))"
  grep -E "DONE |FAIL |test result" "$HOME/b2_$tag.log" | tail -3
  echo "timeout_hits=$(grep -c 'XYCE_TIMEOUT' "$HOME/b2_$tag.log")"
  echo "NA_rows=$(grep -rh 'true_delay=NA' "$V/$tag" 2>/dev/null | wc -l)"
  echo "na_err_kinds=$(grep -rho 'error=[^,]*' "$V/$tag" 2>/dev/null | sort | uniq -c | head -3)"
}
an() {
  tag="$1"
  echo "=== AN $tag ==="
  python3 "$HOME/-project/scripts/diag/_shadow_analyze.py" --root "$V/$tag" > "$HOME/b2an_$tag.log" 2>&1
  rc=$?
  echo "an_rc=$rc"
  head -12 "$HOME/b2an_$tag.log"
  grep -E "前置校验失败" "$HOME/b2an_$tag.log" | head -4
}
mk t TL_ONLY=level4/ADD4_OVF XYCE_BIN=$HOME/slowsim.sh TL_MAX_STEPS=2 TL_MAX_WALL_S=300 XYCE_TIMEOUT_S=1
mk n TL_ONLY=level0/AND2 TL_MAX_WALL_S=21600 XYCE_TIMEOUT_S=1800
mk w TL_ONLY=level0/AND2 TL_MAX_WALL_S=120 XYCE_TIMEOUT_S=1800
an n
an w
echo "=== bfval2 all done ==="
