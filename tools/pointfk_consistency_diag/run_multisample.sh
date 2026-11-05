#!/bin/bash
# Multi-sample selection test (model A only, N joint samples/window). Reuses run_one.sh via env. Results: $DIAG_RESULTS.
# usage: run_multisample.sh [NSAMPLES=8]
MY=$(cd "$(dirname "$0")" && pwd); export DIAG_NSAMPLES=${1:-8} DIAG_ORIG_SHARDED=1
export DIAG_RESULTS=${DIAG_RESULTS:-/data/shichaojian/runs/consistency_multisample_20261014}; RES=$DIAG_RESULTS
mkdir -p $RES/logs; PY=/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python
echo "== SMOKE (2 windows, N=2, 1 GPU) $(date)"
DIAG_NSAMPLES=2 DIAG_RESULTS=$RES/smoke bash $MY/run_one.sh A 0 0 1 1 2 > $RES/logs/smoke.log 2>&1
tail -n 3 $RES/logs/smoke.log; [ $(ls $RES/smoke/preds/A/*.npz 2>/dev/null | wc -l) -ge 2 ] || { echo "SMOKE FAILED"; exit 1; }
echo "== FULL N=$DIAG_NSAMPLES $(date)"
for g in 0 1 2 3 4 5 6 7; do
  bash $MY/run_one.sh A $g $g 8 1 0 > $RES/logs/full_A_$g.log 2>&1 &
  sleep 15; done; wait
echo "== ANALYSIS $(date)"
CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH="" PYTHONPATH=$MY $PY $MY/analyze_multisample.py $RES/preds/A $RES --single /data/shichaojian/runs/consistency_diag_20261014/preds/A > $RES/logs/analysis.log 2>&1; tail -n 80 $RES/logs/analysis.log
echo "ALL_DONE $(date)"
