#!/usr/bin/env bash

set -x

python -m torch.distributed.launch --master_port=29500 --nproc_per_node=7  main.py --launcher pytorch --config cfgs/MultiScale_models/dvae_preprocess.yaml --exp_name preprocess_region
