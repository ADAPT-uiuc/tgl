#!/bin/bash

#SBATCH --job-name=dist-test
#SBATCH --output=dist-test.out
#SBATCH --error=dist-test.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbzw-delta-gpu
#SBATCH --gpus-per-node=4

export MASTER_PORT=12345
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

export LOGLEVEL=INFO

conda activate tgl

echo "start: $(date)"

cd ..

srun python train_dist_mp.py  --data wiki --config exp/tgat.yml

echo "end $(date)"

exit