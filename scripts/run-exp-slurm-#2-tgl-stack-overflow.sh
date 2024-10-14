#!/bin/bash
#SBATCH --mem=2000g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x8
#SBATCH --time=3:00:00
#SBATCH --account=bbzw-delta-gpu
#SBATCH --job-name=stack-overflow-tgl-2
#SBATCH --output=stack-overflow-tgl-2.out
#SBATCH --error=stack-overflow-tgl-2.err
### GPU options ###
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=2
###SBATCH --gpu-bind=verbose,per_task:1
#SBATCH --gpu-bind=closest     # <- or closest


source ~/.bashrc
conda activate tgl
conda info -e
num_gpus=2

echo "job is starting on `hostname`"
echo "start: $(date)"

for model in TGAT APAN; do
    echo "tgl stack-overflow $model";
    srun python -m torch.distributed.launch --nproc_per_node=3 train_dist.py  --data stack-overflow --config "/u/wanyu/github/tgl/examples/exp/dist/$model.yml" --num_gpus $num_gpus;

    for rank in $(seq 0 $((num_gpus))); do
        srun mv out-stats-$rank.csv "$num_gpus-out-tgl-stack-overflow-$model-$rank.csv";
    done
done

echo "end $(date)"
exit
