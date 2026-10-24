#!/bin/bash
# usage: run_one.sh KEY GPU SHARD NSHARD ORIG [LIMIT]     KEY in A|B|C (unsynced, 10-14) | S25|S30 (synced nolocal, D0 10-24)
KEY=$1; GPU=$2; SHARD=$3; NSH=$4; ORIG=$5; LIMIT=${6:-0}
RUNS=/data/shichaojian/runs; MY=$(cd "$(dirname "$0")" && pwd)
RES=${DIAG_RESULTS:-$RUNS/consistency_diag_20261014}; MAIN=$(cd "$MY/../.." && pwd)
case $KEY in
  A) CK=$(ls -d $RUNS/sim_task27_ptfk_nonlocal_8gpu_40k_20261010/cosmos3_action/action_sft/*/checkpoints/iter_000040000|head -1); VAR=nonlocal ;;
  B) CK=$(ls -d $RUNS/sim_task27_ptfk_local_8gpu_40k_20261010/cosmos3_action/action_sft/*/checkpoints/iter_000040000|head -1); VAR=local ;;
  C) CK=$(ls -d $RUNS/sim_task27_pointfk_nonlocal_20k/cosmos3_action/action_sft/*/checkpoints/iter_000020000|head -1); VAR=nonlocal ;;
  S25) CK=$(ls -d $RUNS/sim_task27_ptfk_nonlocal_sync_8gpu_40k_20261021/cosmos3_action/action_sft/*/checkpoints/iter_000025000|head -1); VAR=nonlocal ;;
  S30) CK=$(ls -d $RUNS/sim_task27_ptfk_nonlocal_sync_8gpu_40k_20261021/cosmos3_action/action_sft/*/checkpoints/iter_000030000|head -1); VAR=nonlocal ;;
  *) exit 2;;
esac
# private copies of the launchers: same env/args, but the trainer module is our eval-only wrapper (repo untouched)
L=$RES/launch_$KEY$SHARD; mkdir -p $L/examples
for f in launch_sim_normalized.sh launch_sim_pointfk_v2.sh _sft_launcher_common.sh; do
  sed -e "s#^ROOT=.*#ROOT=$MAIN#" -e "s#^WORKDIR=.*#WORKDIR=$MAIN#" \
      -e "s#PYTHONPATH=\. \\\\#PYTHONPATH=.:$MY \\\\#" -e "s#-m cosmos_framework.scripts.train#-m diag_entry#" \
      -e "s#\$ROOT/examples/launch_sim_pointfk_v2.sh#$L/examples/launch_sim_pointfk_v2.sh#" \
      -e "s#source examples/_sft_launcher_common.sh#source $L/examples/_sft_launcher_common.sh#" \
      -e "s#bash \"\$ROOT/examples/launch_sim_pointfk_v2.sh\"#bash $L/examples/launch_sim_pointfk_v2.sh#" $MAIN/examples/$f > $L/examples/$f
done
grep -q "diag_entry" $L/examples/_sft_launcher_common.sh || { echo "sed failed"; exit 3; }
cd $MAIN; export LD_LIBRARY_PATH=""
export CUDA_VISIBLE_DEVICES=$GPU NPROC_PER_NODE=1 NNODES=1 MASTER_PORT=$((29700+GPU*3+SHARD%3)) RANK=0 LOCAL_RANK=0 WORLD_SIZE=1
export OUTPUT_ROOT=$RES/evalrun_${KEY}_s$SHARD
export DIAG_OUT=$RES/preds/$KEY DIAG_SHARD=$SHARD DIAG_NSHARD=$NSH DIAG_ORIG=$ORIG DIAG_LIMIT=$LIMIT DIAG_STRIDE=24
export EXTRA_TAIL_OVERRIDES="checkpoint.load_path=$CK checkpoint.load_training_state=false checkpoint.strict_resume=true checkpoint.keys_to_skip_loading=[] trainer.max_iter=1"
mkdir -p $OUTPUT_ROOT $DIAG_OUT
echo "key=$KEY gpu=$GPU shard=$SHARD/$NSH ckpt=$CK"
bash $L/examples/launch_sim_normalized.sh 27_ball $VAR
echo "DIAG_EXIT $?"
