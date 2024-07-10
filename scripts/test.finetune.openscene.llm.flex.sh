#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/Adaptive-LLM-finetune-Openscene-test-FLEX-threshold.yaml \
    --exp_name test\
    --test \
    --ckpt experiments/Adaptive-LLM-finetune-Openscene-FLEX-threshold/MultiScale_models/Exp0032_0626_FLEX_AddSceneLoss_LL3daData_LL3DAVisualPrompt_AddLL3DAPos_LL3DAandSystemPrompt_VisualSpecialToken_FlexThreshold127_From[Scratch]/ckpt-epoch-003.pth
