#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/objectcentric/ObjectCentric-Adaptive-LLM-finetune-Openscene-test-hm3dqa.yaml \
    --exp_name Exp0122_0801_LeoModel_ObjectCaptionWoVP_SeqLen256_WoClipNorm_WSceneLoss_WHdBboxAug_WDiffPrompt_From[Scratch]_Epoch4\
    --ckpt experiments/ObjectCentric-Adaptive-LLM-finetune-Openscene-HD/objectcentric/Exp0122_0801_LeoModel_ObjectCaptionWoVP_SeqLen256_WoClipNorm_WSceneLoss_WHdBboxAug_WDiffPrompt_From[Scratch]/ckpt-epoch-003.pth\
    --test
