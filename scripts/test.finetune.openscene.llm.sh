#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/Adaptive-LLM-finetune-Openscene-test-hm3dqa-MoreToken.yaml \
    --exp_name Exp0084_0713_SeqLen256_WoClipNorm_WSceneLoss_WHdBboxAug_WDiffPrompt_From[Scratch]_Epoch4\
    --ckpt experiments/Adaptive-LLM-finetune-Openscene-HD-MoreToken/MultiScale_models/Exp0084_0713_SeqLen256_WoClipNorm_WSceneLoss_WHdBboxAug_WDiffPrompt_From[Scratch]/ckpt-epoch-003.pth\
    --test
