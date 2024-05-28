#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/Adaptive-LLM.yaml \
    --exp_name EqualData_ObjCls \
    --ckpt experiments/Point-BERT/MultiScale_models/ckpt-best.pth
    # --resume