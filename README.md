This is the TGL code from [github][tgl], slightly modified with a bit of
cleanup and timing code for experimentation purposes.

## Getting Started

Create and activate a python environment:

```
$ conda create -n tgl python=3.7
$ conda activate tglite
```

Install dependencies that have CUDA versions (`dgl` has been updated since
older version is not available, and `torch` is updated for fairer comparison):

```
$ pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install torch-scatter==2.1.0+pt112cu116 -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
$ pip install dgl==1.0.1+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
```

Then install the rest of the dependencies in `requirements.txt`.

## Running Experiments

To run the models, first download the datasets by running the `down.sh` script.
There's a lot of data so you might want to start by commenting out `gdelt` and
`mag`. Then, use the configs in `exp/` directory, e.g.:

```
$ python train.py --data wiki --config exp/tgat-l2-n10-b600-fgpu.yml
```

[tgl]: https://github.com/amazon-science/tgl
