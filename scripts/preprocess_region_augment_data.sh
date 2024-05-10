#!/usr/bin/env bash

set -x

python main.py --config cfgs/MultiScale_models/dvae_preprocess.yaml --exp_name preprocess_region
