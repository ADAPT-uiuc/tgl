#!/bin/bash
#SBATCH --mem=2000g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x8
#SBATCH --time=4:00:00
#SBATCH --account=bbzw-delta-gpu
#SBATCH --job-name=netflix-tgl-8
#SBATCH --output=netflix-tgl-8.out
#SBATCH --error=netflix-tgl-8.err
### GPU options ###
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=8
###SBATCH --gpu-bind=verbose,per_task:1
#SBATCH --gpu-bind=closest     # <- or closest


source ~/.bashrc
conda activate tgl
conda info -e
num_gpus=8

echo "job is starting on `hostname`"
echo "start: $(date)"

for model in TGAT APAN TGN JODIE; do
    echo "tgl netflix $model";
    srun python -m torch.distributed.launch --nproc_per_node=9 train_dist.py  --data netflix --config "/u/wanyu/github/tgl/examples/exp/dist/$model.yml" --num_gpus $num_gpus;

    for rank in $(seq 0 $((num_gpus))); do
        srun mv out-stats-$rank.csv "$num_gpus-out-tgl-netflix-$model-$rank.csv";
    done
done

echo "end $(date)"
exit
