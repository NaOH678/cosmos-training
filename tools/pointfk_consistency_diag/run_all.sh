#!/bin/bash
# Results: $DIAG_RESULTS (default /data/shichaojian/runs/consistency_diag_20261014). Code: this repo (launchers, model, consistency module).
# one 8-GPU job: smoke (2 windows, 1 GPU per model) -> full run (A: 3 shards, B: 3, C: 2) -> analysis
MY=$(cd "$(dirname "$0")" && pwd); RES=${DIAG_RESULTS:-/data/shichaojian/runs/consistency_diag_20261014}
mkdir -p $RES/logs; cd $MY; PY=/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python
echo "== SMOKE $(date)"
for k in 0:A 1:B 2:C; do g=${k%%:*}; m=${k##*:}; bash $MY/run_one.sh $m $g 0 1 1 2 > $RES/logs/smoke_$m.log 2>&1 & done; wait
for m in A B C; do grep -c "\[diag\] saved" $RES/logs/smoke_$m.log; tail -n 2 $RES/logs/smoke_$m.log; done
for m in A B C; do [ $(ls $RES/preds/$m/*.npz 2>/dev/null | wc -l) -ge 2 ] || { echo "SMOKE FAILED $m"; exit 1; }; done
echo "== FULL $(date)"
i=0
for spec in A:0:3 A:1:3 A:2:3 B:0:3 B:1:3 B:2:3 C:0:2 C:1:2; do IFS=: read m s n <<< "$spec"
  o=0; [ $s = 0 ] && o=1
  bash $MY/run_one.sh $m $i $s $n $o 0 > $RES/logs/full_${m}_$s.log 2>&1 &
  i=$((i+1)); sleep 15; done; wait
echo "== ANALYSIS $(date)"
CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH="" PYTHONPATH=$MY $PY $MY/analyze.py $RES/preds $RES > $RES/logs/analysis.log 2>&1; tail -n 60 $RES/logs/analysis.log
echo "ALL_DONE $(date)"
