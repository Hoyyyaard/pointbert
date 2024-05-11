#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main.py \
    --launcher pytorch --sync_bn \
    --config cfgs/MultiScale_models/dvae.yaml \
    --exp_name multiscale_dvae_test \
    # --resume