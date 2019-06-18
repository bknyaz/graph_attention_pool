# Intro

This repository contains code to generate data and reproduce experiments from the extended version of our paper:

[Boris Knyazev, Graham W. Taylor, Mohamed R. Amer. Understanding attention in graph neural networks](https://arxiv.org/abs/1905.02850), presented as a contributed talk at [ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019](https://rlgm.github.io/cfp/).

*The paper will be updated soon!*

*The code to train and evaluate models is also coming soon!*

## Tasks & Datasets

1. We design two synthetic graph tasks, COLORS and TRIANGLES, in which we predict the number of green nodes and the number of triangles respectively.

2. We also experiment with the [MNIST](http://yann.lecun.com/exdb/mnist/) image classification dataset, which we preprocess by extracting superpixels - a more natural way to feed images to a graph. We denote this dataset as MNIST-75sp.

3. We validate our weakly-supervised approach on three common graph classification benchmarks: COLLAB, PROTEINS and D&D.

For COLORS, TRIANGLES and MNIST we know ground truth attention for nodes, which allows us to study graph neural networks with attention in depth.

<figure> <img src="data/datasets.png" height="400"><figcaption></figcaption></figure>


## Data generation

To generate all data using a single command, run ```./prepare_data.sh```.

All generated/downloaded ata will be stored in the local ```./data``` directory.
It can take 1-2 hours to prepare all data.

Alternatively, you can generate data for each task as described below.

In case of any issues with running these scripts, they can be downloaded from [here](https://drive.google.com/drive/folders/1Prc-n9Nr8-5z-xphdRScftKKIxU4Olzh?usp=sharing).

### COLORS
To generate training, validation and test data for our **Colors** dataset with different dimensionalities
(in the current version of the paper ```dim```=3):

```for dim in 3 8 16 32; do python generate_data.py --dim $dim; done```

### MNIST-75sp
To generate training and test data for our MNIST-75sp dataset using 4 CPU threads:

```for split in train test; do python extract_superpixels.py -s $split -t 4; done```

### CIFAR-10-150sp [optionally]
Our code also supports [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), so that training and test data can be generated as following:

```for split in train test; do python extract_superpixels.py -D cifar10 -c 10 -n 150 -s $split -t 4; done```

## Data visualization
Once datasets are generated or downloaded, you can use the following IPython notebooks to load and visualize data:

[COLORS and TRIANGLES](graphs_visualize.ipynb) and [MNIST-75sp and CIFAR-10-150sp](superpixels_visualize.ipynb).

# Reference

Please cite our paper if you use our data or code.

```
@inproceedings{knyazev2019understanding,
  author = {Boris Knyazev and Graham Taylor and Mohamed Amer},
  title = {Understanding Attention in Graph Neural Networks},
  booktitle = {International Conference on Learning Representations (ICLR) Workshop on Representation Learning on Graphs and Manifolds},
  year = 2019,
  pdf = {http://arxiv.org/abs/1905.02850}
}
```
