#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/Adaptive-LLM.yaml \
    --exp_name 0530_Batch_EqualData_From[Uclip2] \
    --ckpt ckpts/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt \
    # --resume