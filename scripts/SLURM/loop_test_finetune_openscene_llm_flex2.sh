NUM_GPUS_PER_NODE=${2:-6}
NUM_NODES=${3:-5}
JOB_ID=${4:-"allm"}
LOOP_COUNTER=0
SCRIPT=${1:-"/gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/pointbert/scripts/SLURM/test.openscene.finetune.llm.flex2.sh"}

srun -J allm --gres=gpu:$NUM_GPUS_PER_NODE -N $NUM_NODES  --mem=500G --time 06:00:00 --pty bash $SCRIPT $NUM_GPUS_PER_NODE $NUM_NODES $JOB_ID

