#!/bin/bash
# FDM/IDM GT-injection probe + video perturbation (D0). DIAG_KEY=A|S25|S30, DIAG_NGPU=8. Reuses run_one.sh. usage: run_probe.sh [smoke|full|all]
MY=$(cd "$(dirname "$0")" && pwd); export DIAG_PROBE=1 DIAG_ORIG_SHARDED=1
export DIAG_RESULTS=${DIAG_RESULTS:-/data/shichaojian/runs/fdm_idm_probe_20261014}; RES=$DIAG_RESULTS
KEY=${DIAG_KEY:-A}; export DIAG_KEY=$KEY
PRIOR=/data/shichaojian/runs/consistency_diag_20261014
PY=/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python
mkdir -p $RES/logs
: > $RES/decode_list.txt   # 10-24: no per-window table any more; full runs decode nothing
export DIAG_DECODE_LIST=$RES/decode_list.txt DIAG_VIDEO_DIR=$RES/videos
MODE=${1:-all}
if [ $MODE != full ]; then
  echo "== SMOKE $(date)"
  echo "*" > $RES/decode_all.txt
  DIAG_DECODE_LIST=$RES/decode_all.txt DIAG_VIDEO_DIR=$RES/smoke/videos DIAG_RESULTS=$RES/smoke bash $MY/run_one.sh $KEY ${SMOKE_GPU:-0} 0 1 0 2 > $RES/logs/smoke.log 2>&1
  tail -n 4 $RES/logs/smoke.log; [ $(ls $RES/smoke/preds/$KEY/*.npz 2>/dev/null | wc -l) -ge 2 ] || { echo "SMOKE FAILED"; exit 1; }
  [ $MODE = smoke ] && exit 0
fi
echo "== FULL $(date)"
NG=${DIAG_NGPU:-8}
for g in $(seq 0 $((NG-1))); do
  bash $MY/run_one.sh $KEY $g $g $NG 1 0 > $RES/logs/full_A_$g.log 2>&1 &
  sleep 15; done; wait
echo "== ANALYSIS $(date)"
CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH="" PYTHONPATH=$MY $PY $MY/analyze_probe.py $RES/preds/$KEY $RES /nonexistent > $RES/logs/analysis.log 2>&1; tail -n 5 $RES/logs/analysis.log
echo "ALL_DONE $(date)"
