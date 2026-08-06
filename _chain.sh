#!/bin/bash
# 链式启动: 跑完 hard5 → hard5w2 → hard10w2
# 用法: bash _chain.sh

RUNS="hard5 hard5w2 hard10w2"
PREV=""
for V in $RUNS; do
  echo "[$(date)] Starting $V..."
  if [ -n "$PREV" ]; then
    # 等上一个结束
    echo "  Waiting for $PREV to finish..."
    while ! grep -q "^SUMMARY" ~/project-107-$PREV/train107$PREV.log 2>/dev/null; do
      sleep 60
    done
    echo "  $PREV finished."
  fi
  cd ~/exp107 && git pull 2>/dev/null
  bash ~/exp107/setup_exp.sh $V
  PREV=$V
done
echo "[$(date)] All done."
