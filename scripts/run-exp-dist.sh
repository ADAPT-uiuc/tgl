#!/usr/bin/env bash
#
# Run experiments for distributed training.
#

repo="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$repo"

echo "start: $(date)"

num_gpus=4

for data in gdelt mag; do
  for model in TGN APAN; do
    echo "tgl $data $model";
    python -m torch.distributed.launch --nproc_per_node=$((num_gpus+1)) \
    train_dist.py --data $data --config "exp/dist/$model.yml" \
    --num_gpus $num_gpus --data_path /shared/;
    for rank in $(seq 0 $num_gpus); do
      mv out-stats-$rank.csv "$num_gpus-out-tgl-$data-$model-$rank.csv";
    done
    echo;
    echo "time: $(date)"
    echo;
  done
done


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

# model=TGAT
# python -m torch.distributed.launch --nproc_per_node=$((num_gpus+1)) \
# train_dist.py --data wiki --config "exp/dist/$model.yml" \
# --num_gpus $num_gpus --data_path /shared/

echo "end: $(date)"