#!/usr/bin/env bash

set -x


CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/MultiScale_models/dvae_preprocess.yaml --exp_name preprocess_region
