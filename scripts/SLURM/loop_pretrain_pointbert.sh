NUM_GPUS_PER_NODE=${2:-6}
NUM_NODES=${3:-8}
JOB_ID=${4:-"dvae"}
LOOP_COUNTER=0
SCRIPT=${1:-"/gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/pointbert/scripts/SLURM/train.multiscale.pointbert.sh"}

while true; do
    echo "Loop counter: $LOOP_COUNTER"
    srun -J dvae --gres=gpu:$NUM_GPUS_PER_NODE -N $NUM_NODES  --mem=500G --time 06:00:00 --pty bash $SCRIPT $NUM_GPUS_PER_NODE $NUM_NODES $JOB_ID
    sleep 10
    LOOP_COUNTER=$((LOOP_COUNTER+1))
done
