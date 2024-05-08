#!/usr/bin/env bash

set -x

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py \
    --config cfgs/MultiScale_models/dvae.yaml \
    --exp_name multiscale_dvae \
    --val_freq 10 \