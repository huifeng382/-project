#!/bin/bash
# 17.6.0 positive validation: 3 runs. Own out root + own cache dir per run. ASCII only.
# Run on server in ~/NetlistOpt. Writes ~/bf_<tag>.log and prints one summary line per
# field. Usage: bash ~/bfval.sh
set -u
cd "$HOME/NetlistOpt" || exit 1
export OMP_NUM_THREADS=6
V="$PWD/temp_sim_test/bfval"
K="$PWD/temp_sim_cache_bfval"
mkdir -p "$V" "$K"
B="GNN_SHADOW=1 TL_BESTFIRST=1 TL_MAX_STEPS=40"
T="cargo test --release --test tl_opt_gnn_batch -- --nocapture --ignored"
echo "V=$V"
echo "K=$K"
echo "B=$B"
echo "T=$T"
echo "--- previous bf_a.log error lines (saved as bf_a.log.old) ---"
cp -f "$HOME/bf_a.log" "$HOME/bf_a.log.old" 2>/dev/null
grep -E "^error" "$HOME/bf_a.log.old" 2>/dev/null | head -5
mk() {
  tag="$1"; shift
  echo "=== RUN $tag ==="
  echo "env: $*"
  s=$(date +%s)
  env $B "$@" SHADOW_OUT_BASE="$V/$tag" XYCE_CACHE_DIR="$K/$tag" $T > "$HOME/bf_$tag.log" 2>&1
  rc=$?
  echo "rc=$rc wall_s=$(( $(date +%s) - s ))"
  grep -E "DONE |test result" "$HOME/bf_$tag.log" | tail -3
  echo "err_lines=$(grep -cE '^error' "$HOME/bf_$tag.log")"
  echo "timeout_hits=$(grep -c 'XYCE_TIMEOUT' "$HOME/bf_$tag.log")"
  echo "NA_rows=$(grep -rh 'true_delay=NA' "$V/$tag" 2>/dev/null | wc -l)"
  echo "na_err_kinds=$(grep -rho 'error=[^,]*' "$V/$tag" 2>/dev/null | sort | uniq -c | head -3)"
}
mk d TL_ONLY=level4/ADD4_OVF TL_MAX_WALL_S=5 XYCE_TIMEOUT_S=1800
mk b TL_ONLY=level4/ADD4_OVF TL_MAX_WALL_S=21600 XYCE_TIMEOUT_S=1
mk a TL_ONLY=level0/AND2 TL_MAX_WALL_S=21600 XYCE_TIMEOUT_S=1800
echo "=== bfval all done ==="
