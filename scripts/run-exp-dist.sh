#!/usr/bin/env bash
#
# Run experiments for distributed training.
#

repo="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$repo"

echo "start: $(date)"

num_gpus=4
model=TGAT

# for model in APAN JODIE TGAT TGN; do
#   echo "tgl wiki $model";
#   python -m torch.distributed.launch --nproc_per_node=$((num_gpus+1)) \
#   train_dist.py --data wiki --config "./config/dist/$model.yml" \
#   --num_gpus $num_gpus --data_path /shared/;
#   for rank in $(seq 0 $num_gpus); do
#     mv out-stats-$rank.csv "out-tgl-wiki-$model-$rank.csv";
#   done
#   echo;
#   echo "time: $(date)"
#   echo;
# done


# for model in APAN JODIE TGAT TGN; do
#   echo "tgl gdelt $model";
#   python -m torch.distributed.launch --nproc_per_node=$((num_gpus+1)) \
#   train_dist.py --data gdelt --config "./config/dist/$model.yml" \
#   --num_gpus $num_gpus --data_path /shared/;
#   for rank in $(seq 0 $num_gpus); do
#     mv out-stats-$rank.csv "out-tgl-gdelt-$model-$rank.csv";
#   done
#   echo;
#   echo "time: $(date)"
#   echo;
# done

python -m torch.distributed.launch --nproc_per_node=$((num_gpus+1)) \
train_dist.py --data wiki --config "./config/dist/$model.yml" \
--num_gpus $num_gpus --data_path /shared/

echo "end: $(date)"