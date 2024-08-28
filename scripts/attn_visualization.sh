#!/usr/bin/env bash

set -x
PORT=12324
ADDR=127.0.0.1
NGPUS=1

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher none  \
    --config cfgs/MultiScale_models/objectcentric/ObjectCentric-Adaptive-LLM-finetune-Openscene-test-FLEX-threshold-hm3dqa-vis.yaml \
    --exp_name Exp0110_0726_LeoModel_DenseToken4_FixOrderBug_WHdAugBbox_DiffPrompt_FlexWarmUp20_Threshold127_From[Scratch]_Epoch4 \
    --ckpt experiments/ObjectCentric-Adaptive-LLM-finetune-Openscene-FLEX-threshold-HD/objectcentric/Exp0110_0726_LeoModel_DenseToken4_FixOrderBug_WHdAugBbox_DiffPrompt_FlexWarmUp20_Threshold127_From[Scratch]/ckpt-epoch-003.pth\
    --test \
    --visualization_attn