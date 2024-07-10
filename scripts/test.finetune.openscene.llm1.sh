#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

export NORM=True

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/Adaptive-LLM-finetune-Openscene-test-hm3dqa.yaml \
    --exp_name Exp0072_0709_SeqLen256_WoClipNorm_WSceneLoss_ReproduceExp0030_From[Scratch]_Epoch3\
    --ckpt experiments/Adaptive-LLM-finetune-Openscene/MultiScale_models/Exp0072_0709_SeqLen256_WoClipNorm_WSceneLoss_ReproduceExp0030_From[Scratch]/ckpt-epoch-002.pth\
    --test
