#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/Adaptive-LLM-Openscene-test.yaml \
    --exp_name 0604_Pretrain_LL3daData_Batch_AddPos_DetPrompt_FP32_From[0604_Pretrain_Batch_TokenMask_AddPos_DetPrompt_From[Openscene]]_Epoch5 \
    --ckpt experiments/Adaptive-LLM-Openscene/MultiScale_models/0604_Pretrain_LL3daData_Batch_AddPos_DetPrompt_FP32_From[0604_Pretrain_Batch_TokenMask_AddPos_DetPrompt_From[Openscene]]/ckpt-epoch-004.pth\
    --test
