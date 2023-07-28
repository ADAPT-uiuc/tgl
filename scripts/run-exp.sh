#!/usr/bin/env bash
#
# Run experiments for standard benchmarks.
#

repo="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$repo"

echo "start: $(date)"

for data in wiki mooc reddit lastfm; do
  for model in apan jodie tgat tgn; do
    echo "tgl $data $model";
    python train.py --data "$data" --config "exp/$model.yml";
    mv out-stats.csv "out-tgl-$data-$model.csv";
    echo;
    echo "time: $(date)"
    echo;
    python train.py --data "$data" --config "exp/$model-all-gpu.yml";
    mv out-stats.csv "out-tgl-allgpu-$data-$model.csv";
    echo;
    echo "time: $(date)"
    echo;
  done
done

echo "end: $(date)"
