#!/bin/bash
#SBATCH --mem=2000g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x8
#SBATCH --time=2:00:00
#SBATCH --account=bbzw-delta-gpu
#SBATCH --job-name=gdelt-apan-tgl-1
#SBATCH --output=gdelt-apan-tgl-1.out
#SBATCH --error=gdelt-apan-tgl-1.err
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
###SBATCH --gpu-bind=verbose,per_task:1
#SBATCH --gpu-bind=closest     # <- or closest

# source deactivate
# conda deactivate
# module purge
# module load anaconda3_gpu
# module list
source ~/.bashrc
conda activate tgl
conda info -e


echo "job is starting on `hostname`"

echo "start: $(date)"

srun python train.py  --data stack-overflow --config "/u/wanyu/github/tgl/examples/exp/single/apan-gdelt.yml";
srun mv out-stats.csv "out-tgl-stack-overflow-apan.csv";

echo "end $(date)"

exit
