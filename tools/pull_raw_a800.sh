#!/bin/bash
# 10-29 (user: 26 任务混训): pull the 4090-downloaded replay HDF5s from the bucket into the layout of
# tools/cache_bench2dex_multitask_latents.py (<raw>/<short>/dataset/<task>/replay-generalization). The VAE tool itself
# re-checks every file's SHA256 against <task>.files.json, so both pipelines read identical bytes.
BK=aliyun-4090-oss:pjlab-bjpai-manip/shichaojian/bench2dex_raw; RAW=/data/shichaojian/raw_data/bench2dex
L=/data/shichaojian/raw_data/bench2dex/logs; touch $L/pulled.txt
short() { n=${1%%_*}; w=$(echo $1 | cut -d_ -f2); echo task${n}_$w; }
while true; do
  for t in $(bash -ic "rclone lsf $BK --dirs-only" 2>/dev/null | tr -d /); do
    grep -qx "$t" $L/pulled.txt && continue
    bash -ic "rclone lsf $BK/$t/" 2>/dev/null | grep -qx UPLOAD_DONE || continue
    D=$RAW/$(short $t)/dataset/$t/replay-generalization
    bash -ic "rclone copy $BK/$t/replay-generalization $D --transfers 8" >/dev/null 2>&1 \
      && bash -ic "rclone check --size-only $BK/$t/replay-generalization $D" >/dev/null 2>&1 \
      && echo "$t" >> $L/pulled.txt && echo "PULLED $t $(date -u +%H:%M)"
  done
  [ $(wc -l < $L/pulled.txt) -ge 22 ] && { echo PULL_ALL_DONE; break; }
  sleep 180
done
