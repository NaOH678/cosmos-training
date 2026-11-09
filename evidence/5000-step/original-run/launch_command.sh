#!/usr/bin/env bash
set -euo pipefail
cd /mnt/lvminrui/workspace/mrlv/wam-910b-8x-5000-20261105T131003Z
nohup setsid bash ./run_training.sh > ./console.log 2>&1 < /dev/null &
printf '%s\n' "$!" > detached.pid
