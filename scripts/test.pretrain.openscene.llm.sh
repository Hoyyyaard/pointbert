#!/usr/bin/env bash

set -x
PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch  \
    --config cfgs/MultiScale_models/Adaptive-LLM-Openscene-test.yaml \
    --exp_name PExp0003_0613_TokenMask_AddPos_DetPrompt_Equal61kOSRData_Sybn_From[Openscene]\
    --ckpt experiments/Adaptive-LLM-Openscene/MultiScale_models/PExp0003_0613_TokenMask_AddPos_DetPrompt_Equal61kOSRData_Sybn_From[Openscene]/ckpt-last.pth \
    --test
