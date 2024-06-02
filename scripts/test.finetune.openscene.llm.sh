#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/Adaptive-LLM-Openscene-test.yaml \
    --exp_name 0601_Pretrain_EqualData_Batch_TokenMask_AddPos_From[Openscene]_Epoch2 \
    --ckpt experiments/Adaptive-LLM-Openscene/MultiScale_models/0601_Pretrain_EqualData_Batch_TokenMask_AddPos_From[Openscene]/ckpt-last.pth \
    --test