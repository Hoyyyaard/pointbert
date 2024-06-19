#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/Adaptive-LLM-finetune-Openscene-test.yaml \
    --exp_name Exp0008_0613_LL3daData_TokenMask_AddPos_DetPrompt_FP32_SyBN_From[PExp0003]_Epoch1\
    --ckpt experiments/Adaptive-LLM-finetune-Openscene/MultiScale_models/Exp0008_0613_LL3daData_TokenMask_AddPos_DetPrompt_FP32_SyBN_From[PExp0003]/ckpt-epoch-000.pth \
    --test
