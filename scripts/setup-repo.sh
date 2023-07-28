#!/usr/bin/env bash
#
# Script to setup environment for this repo, expects conda to be available.
#

conda_dir="$HOME/.conda"
conda_bin="$conda_dir/bin/conda"
repo="$(cd "$(dirname "$0")"; cd ..; pwd)"

echo
echo ">> setting up environment"
echo

"$conda_bin" create -n tgl python=3.7
"$conda_bin" activate tgl

echo
echo ">> installing python packages"
echo

pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter==2.1.0+pt112cu116 -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
pip install dgl==1.0.1+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
pip install -r requirements.txt

echo
echo ">> compiling c++ extension"
echo

python setup.py build_ext --inplace

cd "$repo"
if [[ -d "DATA/wiki-talk" && ! -f "DATA/wiki-talk/ext_full.npz" ]]; then
  echo
  echo ">> preparing datasets"
  echo
  python gen_graph.py --data wiki-talk --add_reverse
fi

echo
echo ">> cleaning up"
echo

make clean
conda clean -a -y
pip cache purge
rm -rf ~/.cache

echo
echo ">> done! please restart your shell session"
