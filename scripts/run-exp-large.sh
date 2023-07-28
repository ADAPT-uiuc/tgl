#!/usr/bin/env bash
#
# Run experiments for larger benchmarks.
#

repo="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$repo"

echo "start: $(date)"

for model in apan jodie tgat tgn; do
  echo "tgl wiki-talk $model";
  python train.py --data wiki-talk --config "exp/$model.yml";
  mv out-stats.csv "out-tgl-wiki-talk-$model.csv";
  echo;
  echo "time: $(date)"
  echo;
done

for model in apan jodie tgat tgn; do
  echo "tgl gdelt $model";
  python train.py --data gdelt --config "exp/$model-gdelt.yml";
  mv out-stats.csv "out-tgl-gdelt-$model.csv";
  echo;
  echo "time: $(date)"
  echo;
done

echo "end: $(date)"
