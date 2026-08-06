#!/bin/bash
# 顺序解冻: hard5 跑完 → 恢复 hard5w2 → 跑完 → 恢复 hard10w2
# 不动已在跑的 hard5，不杀进程，不丢 epoch

NEXT=("hard5w2" "hard10w2")
for V in "${NEXT[@]}"; do
  echo "[$(date)] Waiting for previous to finish..."
  PREV="${V//w2/}"
  PREV="${PREV//5/5}"
  # 找上一个变体的 log
  if [ "$V" = "hard5w2" ]; then PREV_LOG=~/project-107-hard5/train107hard5.log; fi
  if [ "$V" = "hard10w2" ]; then PREV_LOG=~/project-107-hard5w2/train107hard5w2.log; fi
  while [ ! -f "$PREV_LOG" ] || ! grep -q "^SUMMARY" "$PREV_LOG" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date)] Unfreezing $V..."
  ps -eo pid,args | grep "project-107-$V" | grep -v grep | awk '{print $1}' | xargs -r kill -CONT
done
echo "[$(date)] All done."
