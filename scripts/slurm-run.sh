#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbzw-delta-gpu
#SBATCH --gpus-per-node=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

conda activate tgl

echo "start: $(date)"

cd ..

srun python train.py  --data wiki --config exp/tgat.yml

echo "end $(date)"

exit
