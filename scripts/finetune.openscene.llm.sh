#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/Adaptive-LLM-finetune-Openscene-FLEX-threshold-HD.yaml \
    --exp_name debug \
    # --ckpt experiments/Adaptive-LLM-Openscene/MultiScale_models/0530_Pretrain_EqualData_Batch_From[Openscene]/ckpt-last.pth