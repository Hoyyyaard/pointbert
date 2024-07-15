#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

export NORM=True

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/Adaptive-LLM-finetune-Openscene-test.yaml \
    --exp_name Exp0073_0710_SeqLen256_WoClipNorm_WSceneLoss_WHdBboxAug_WDiffPrompt_ReproduceExp0035_From[Scratch]_Epoch3_Beam4\
    --ckpt experiments/Adaptive-LLM-finetune-Openscene-HD/MultiScale_models/Exp0073_0710_SeqLen256_WoClipNorm_WSceneLoss_WHdBboxAug_WDiffPrompt_ReproduceExp0035_From[Scratch]/ckpt-epoch-002.pth\
    --test
