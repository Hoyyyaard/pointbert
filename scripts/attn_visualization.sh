#!/usr/bin/env bash

set -x
PORT=12324
ADDR=127.0.0.1
NGPUS=1

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher none  \
    --config cfgs/MultiScale_models/qformer/Qformer-Adaptive-LLM-finetune-Openscene-test-FLEX-QueryProb-threshold-HD-hm3dqa-vis.yaml \
    --exp_name Exp0107_0726_WHdAugBbox_DiffPrompt_FlexWarmUp-1_Threshold127_QueryProb_QformerAttnLayer0_From[LL3DA]_Epoch4 \
    --ckpt experiments/Qformer-Adaptive-LLM-finetune-Openscene-FLEX-QueryProb-threshold-HD/qformer/Exp0107_0726_WHdAugBbox_DiffPrompt_FlexWarmUp-1_Threshold127_QueryProb_QformerAttnLayer0_From[LL3DA]/ckpt-epoch-003.pth\
    --test \
    --visualization_attn