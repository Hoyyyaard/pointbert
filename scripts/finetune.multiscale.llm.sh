#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch --sync_bn \
    --config cfgs/MultiScale_models/Adaptive-LLM-finetune.yaml \
    --exp_name debug \
    --ckpt experiments/Adaptive-LLM/MultiScale_models/0521_MultiScale_LLM_Pretrain/ckpt-last.pth \
    # --resume