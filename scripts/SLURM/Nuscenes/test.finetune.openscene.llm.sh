#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/nuscenes/Adaptive-LLM-finetune-Openscene-test.yaml \
    --exp_name Exp0175_0823_Token64_400k_Pts10k_From[Scratch]_Epoch3_80k\
    --test \
    --ckpt experiments/Adaptive-LLM-finetune-Openscene/nuscenes/Exp0175_0823_Token64_400k_Pts10k_From[Scratch]/ckpt-epoch-002.pth
