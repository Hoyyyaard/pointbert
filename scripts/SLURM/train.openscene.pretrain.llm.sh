#!/usr/bin/env bash

set -x

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

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
    CMD="/gpfs/u/home/LMCG/LMCGljnn/scratch/miniconda3-ppc64le/envs/ll3da/bin/python -u -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$ip --node_rank=$NODE_RANK"
else
    CMD="/gpfs/u/home/LMCG/LMCGljnn/scratch/miniconda3-ppc64le/envs/ll3da/bin/python -u -m torch.distributed.launch  --nproc_per_node=$NUM_GPUS_PER_NODE --master_port=$MASTER_PORT"
fi


cd /gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/pointbert
    $CMD  main_ALLM.py \
    --launcher slurm \
    --config cfgs/MultiScale_models/Adaptive-LLM-Openscene.yaml \
    --exp_name 0530_Pretrain_EqualData_Batch_From[Openscene] \
    --resume

