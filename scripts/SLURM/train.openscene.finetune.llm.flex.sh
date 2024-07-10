#!/usr/bin/env bash

set -x

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HOME=/gpfs/u/home/LMCG/LMCGljnn/scratch
RANDOM=$$
DIV=1000
OFFSET=24000
MASTER_PORT=$(($RANDOM%$DIV+$OFFSET))
NODE_RANK=${SLURM_PROCID}

# if ']' is the last character of the node list
SLURM=${SLURM_NODELIST:0:3}

if [ "${SLURM_NODELIST: -1}" == "]" ]; then
    if [ $SLURM == "npl" ]; then
        # NPL
        ip=${SLURM}${SLURM_NODELIST:4:2}
    else
        # DCS
        ip=${SLURM}${SLURM_NODELIST:4:3}
    fi
    FLAG=1
else
    ip=$SLURM_NODELIST
    FLAG=0
fi

NUM_GPUS_PER_NODE=$1

echo "ip: $ip"
echo "FLAG: $FLAG"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "NODE_RANK: $NODE_RANK"
echo "NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE"
echo "MASTER_PORT: $MASTER_PORT"

if [ $FLAG -eq 1 ]; then
    NUM_NODES=${2:-1}
    CMD="/gpfs/u/home/LMCG/LMCGljnn/scratch/miniconda3-ppc64le/envs/ll3da/bin/python -u -m torch.distributed.launch  --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$ip --node_rank=$NODE_RANK --max_restarts 5"
else
    CMD="/gpfs/u/home/LMCG/LMCGljnn/scratch/miniconda3-ppc64le/envs/ll3da/bin/python -u -m torch.distributed.launch  --nproc_per_node=$NUM_GPUS_PER_NODE --master_port=$MASTER_PORT"
fi

source /gpfs/u/home/LMCG/LMCGljnn/scratch/miniconda3-ppc64le/etc/profile.d/conda.sh
conda activate ll3da
wandb login 2a1e24aab284649d73b3ed748679b099c73ae980

cd /gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/pointbert
    $CMD  main_ALLM.py \
    --launcher slurm \
    --sync_bn \
    --config cfgs/MultiScale_models/Adaptive-LLM-finetune-Openscene-FLEX-threshold.yaml \
    --exp_name Exp0052_0704_WoXyz_WoClickPrompt_FlexLayer08_FlexThreshold127_Warmup20_From[Scratch] \
    --resume
