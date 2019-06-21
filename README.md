# Intro

<<<<<<< HEAD
This repository contains code to generate data and reproduce experiments from:

[Boris Knyazev, Graham W. Taylor, Mohamed R. Amer. Understanding attention in graph neural networks](https://arxiv.org/abs/1905.02850),

presented as a contributed talk at [ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019](https://rlgm.github.io/cfp/).

Some cases might not be supported in our code yet. We are working on completing the code.

<figure> <img src="data/datasets.png" height="400"><figcaption>Datasets</figcaption></figure>


## Data generation

To generate all data using a single command, run ```./prepare_data.sh```.

All generated/downloaded ata will be stored in the local ```./data``` directory.
It can take 1-2 hours to prepare all data.


### COLORS
To generate training, validation and test data for our **Colors** dataset with different dimensionalities
(in the current version of the paper ```dim```=3):

```for dim in 3 8 16 32; do python generate_data.py --dim $dim; done```

### TRIANGLES

To generate training, validation and test for our **Triangles** dataset using 2 CPU threads:

python generate_data.py -D triangles --N_train 30000 --N_val 5000 --N_test 5000 --label_min 1 --label_max 10 --N_max 100 --threads 2

python main.py -D triangles --epochs 100 --lr_decay_step 85,95 --test_batch_size 100 -f 64,64,64 -K 7 --aggregation sum --n_hidden 64 --readout max  --dropout 0 --pool attn_sup_threshold_skip_0.001_0.001 --pool_arch gnn_curr  --results None -d /scratch/ssd/bknyazev/data/random/ | tee results/triangle_sup_curr_0.001_norm_drop_fixed.log

### MNIST

We use standard torchvision.datasets.MNIST for MNIST.

### MNIST-75sp
To generate training and test data for our MNIST-75sp dataset using 4 CPU threads:

```for split in train test; do python extract_superpixels.py -s $split -t 4; done```

### CIFAR-10-150sp [optionally]
Our code also supports CIFAR-10, so that training and test data can be generated as following:

```for split in train test; do python extract_superpixels.py -D cifar10 -c 10 -n 150 -s $split -t 4; done```

In case of any issues with running these scripts, they can be downloaded from:


## Data visualization
Once datasets are generated or downloaded, you can use the following IPython notebooks to load and visualize data:

[COLORS and TRIANGLES](graphs_visualize.ipynb) and [MNIST-75sp and CIFAR-10-150sp](superpixels_visualize.ipynb).


# Pretrained ChebyGIN models

Click on the result to download a model in the PyTorch format.

| Model                 | COLORS | TRIANGLES | MNIST-75sp | COLLAB | PROTEINS | D&D
| --------------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Script to train models | [colors.sh](scripts/colors.sh) | | | | | |
| Global pooling |  |  | | | | |
| Unsupervised attention |  |  | | | | |
| Supervised attention |  |  | | | | |
| Weakly-supervised attention |  |  | | | | | |

./scripts/mnist_75sp.sh

## Other examples of training models

Hyperparameters should be tuned with the ```--validation``` flag.

### COLORS

To run 100 jobs with random seed for the GIN model with unsupervised attention:

```for i in $(seq 1 1 100); do seed=$(( ( RANDOM % 10000 )  + 1 )); python main.py -D colors-3 --epochs 300 --lr_decay_step 280 --test_batch_size 100 -f 64,64 -K 1 --aggregation sum --n_hidden 256 --readout sum  --dropout 0 --pool attn_unsup_threshold_skip_0.03 --pool_arch fc_prev --seed $seed --results None | tee results/colors/seed"$seed".log; done```

See [plot_results.ipynb](plot_results.ipynb) for the example how to visualize results similarly to Figures in the paper.

To run longer training:

### TRIANGLES

GIN with GT attention
 python main.py -D triangles --epochs 100 --lr_decay_step 85,95 --test_batch_size 100 -f 64,64,64 -K 1 --aggregation sum --n_hidden 64 --readout max  --dropout 0 --pool attn_gt_threshold_skip_0_skip --pool_arch gnn_curr  --results None -d /mnt/data/bknyazev/data/graph_data/node_colors_triangles/

python main.py -D triangles --epochs 100 --lr_decay_step 85,95 --test_batch_size 100 -f 64,64,64 -K 7 --aggregation sum --n_hidden 64 --readout max  --dropout 0 --pool attn_sup_threshold_skip_0.01_0.01 --pool_arch gnn_curr  --results None -d /scratch/ssd/bknyazev/data/random/

### MNIST

Training a model on full size MNIST images with supervised attention:

```python main.py -D mnist --epochs 30 --lr_decay_step 20,25 --test_batch_size 200 -f 4,64,512 -K 4 --aggregation mean --n_hidden 0 --readout max --dropout 0.5 --threads 0 --img_features mean,coord --img_noise_levels 0.5,0.75 --kl_weight 100 --pool attn_sup_threshold_skip_skip_0.001 --pool_arch fc_prev``

See [mnist_wsup.sh](mnist_wsup.sh) for an example of training models with different pooling methods.

### MNIST-75sp

## Trained models

### MNIST

### COLLAB, PROTEINS, D&D

Weakly supervised experiments on PROTEINS:

```for i in $(seq 1 1 10); do seed=$(( ( RANDOM % 10000 )  + 1 )); python main.py --seed $seed -D TU --n_nodes 25 --epochs 50 --lr_decay_step 25,35,45 --test_batch_size 100 -f 64,64,64 -K 3 --aggregation mean --n_hidden 0 --readout max --dropout 0.1 --pool attn_sup_threshold_skip_skip_0 --pool_arch fc_prev --results ./results --data_dir ./data/PROTEINS | tee logs/proteins_wsup_5fold_cv_seed"$seed".log; done```

DD:

for folds in 10; do for n_nodes in 200; do for i in $(seq 1 1 10); do seed=$(( ( RANDOM % 10000 )  + 1 )); python main.py --seed $seed -D TU --cv_folds $folds --n_nodes $n_nodes --epochs 50 --lr_decay_step 25,35,45 --test_batch_size 10 -f 64,64,64 -K 3 --aggregation mean --n_hidden 0 --readout max --dropout 0.1 --pool attn_unsup_threshold_skip_skip_0 --pool_arch fc_prev_32 --results /mnt/data/bknyazev/checkpoints/ --data_dir /mnt/data/bknyazev/data/graph_data/DD | tee results/dd_unsup_"$n_nodes"nodes_"$folds"fold_fc32/dd_seed"$seed".log;  done; done; done



=======
This repository contains code to generate data and reproduce experiments from our paper:

[Boris Knyazev, Graham W. Taylor, Mohamed R. Amer. Understanding Attention and Generalization in Graph Neural Networks](https://arxiv.org/abs/1905.02850).

[An earlier short version](https://rlgm.github.io/papers/54.pdf) of our paper was presented as a contributed talk at [ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019](https://rlgm.github.io/cfp/).


| MNIST |  TRIANGLES
|:-------------------------:|:-------------------------:|
| <figure> <img src="data/mnist_animation.gif" height="500"></figure> |  <figure> <img src="data/triangles_animation.gif" height="500"><figcaption></figcaption></figure> |


For MNIST from top to bottom rows:

- input test images with additive Gaussian noise with standard deviation in the range from 0 to 1.4 with step 0.2
- attention coefficients (alpha) predicted by the **unsupervised** model
- attention coefficients (alpha) predicted by the **supervised** model
- attention coefficients (alpha) predicted by our **weakly-supervised** model

For TRIANGLES from top to bottom rows:

- **on the left**: input test graph (with 4-100 nodes) with ground truth attention coefficients, **on the right**: graph obtained by **ground truth** node pooling
- **on the left**: input test graph (with 4-100 nodes) with unsupervised attention coefficients, **on the right**: graph obtained by **unsupervised** node pooling
- **on the left**: input test graph (with 4-100 nodes) with supervised attention coefficients, **on the right**: graph obtained by **supervised** node pooling
- **on the left**: input test graph (with 4-100 nodes) with weakly-supervised attention coefficients, **on the right**: graph obtained by **weakly-supervised** node pooling


Note that during training, our MNIST models have not encountered noisy images and our TRIANGLES models have not encountered graphs larger than with N=25 nodes.


## Tasks & Datasets

1. We design two synthetic graph tasks, COLORS and TRIANGLES, in which we predict the number of green nodes and the number of triangles respectively.

2. We also experiment with the [MNIST](http://yann.lecun.com/exdb/mnist/) image classification dataset, which we preprocess by extracting superpixels - a more natural way to feed images to a graph. We denote this dataset as MNIST-75sp.

3. We validate our weakly-supervised approach on three common graph classification benchmarks: [COLLAB, PROTEINS and D&D](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

For COLORS, TRIANGLES and MNIST we know ground truth attention for nodes, which allows us to study graph neural networks with attention in depth.

<figure> <img src="data/datasets.png" height="400"><figcaption></figcaption></figure>


## Data generation

To generate all data using a single command: ```./scripts/prepare_data.sh```.

All generated/downloaded ata will be stored in the local ```./data``` directory.
It can take 1-2 hours to prepare all data.

Alternatively, you can generate data for each task as described below.

In case of any issues with running these scripts, they can be downloaded from [here](https://drive.google.com/drive/folders/1Prc-n9Nr8-5z-xphdRScftKKIxU4Olzh?usp=sharing).

### COLORS
To generate training, validation and test data for our **Colors** dataset with different dimensionalities:

```for dim in 3 8 16 32; do python generate_data.py --dim $dim; done```

### MNIST-75sp
To generate training and test data for our MNIST-75sp dataset using 4 CPU threads:

```for split in train test; do python extract_superpixels.py -s $split -t 4; done```

## Data visualization
Once datasets are generated or downloaded, you can use the following IPython notebooks to load and visualize data:

[COLORS and TRIANGLES](notebooks/synthetic_graphs_visualize.ipynb), [MNIST](notebooks/superpixels_visualize.ipynb) and
[COLLAB, PROTEINS and D&D](notebooks/graphs_visualize.ipynb).


# Pretrained ChebyGIN models

Generalization results on the test sets for three tasks. Other results are available in the paper.

Click on the result to download a trained model in the PyTorch format.

| Model                 | COLORS-Test-LargeC | TRIANGLES-Test-Large | MNIST-75sp-Test-Noisy
| --------------------- |:-------------:|:-------------:|:-------------:|
| Script to train models | [colors.sh](scripts/colors.sh) | [triangles.sh](scripts/triangles.sh) | [mnist](./scripts/mnist_75sp.sh) |
| Global pooling | [15 ± 7](./checkpoints/checkpoint_colors-3_828931_epoch100_seed0000111.pth.tar) | [30 ± 1](./checkpoints/checkpoint_triangles_658037_epoch100_seed0000111.pth.tar) | [80 ± 12](./checkpoints/checkpoint_mnist-75sp_820601_epoch30_seed0000111.pth.tar)  |
| Unsupervised attention | [11 ± 6](./checkpoints/checkpoint_colors-3_223919_epoch300_seed0000111.pth.tar) | [26 ± 2](./checkpoints//checkpoint_triangles_051609_epoch100_seed0000111.pth.tar)  | [80 ± 23](./checkpoints/checkpoint_mnist-75sp_330394_epoch30_seed0000111.pth.tar)  |
| Supervised attention | [75 ± 17](./checkpoints/checkpoint_colors-3_332172_epoch300_seed0000111.pth.tar) | [48 ± 1](./checkpoints/checkpoint_triangles_586710_epoch100_seed0000111.pth.tar) | [92.3 ± 0.4](./checkpoints/checkpoint_mnist-75sp_139255_epoch30_seed0000111.pth.tar) |
| Weakly-supervised attention | [73 ± 14 ](./checkpoints//checkpoint_colors-3_312570_epoch300_seed0000111.pth.tar) | [30 ± 1](./checkpoints/checkpoint_triangles_230187_epoch100_seed0000111.pth.tar)  | [88.8 ± 4](./checkpoints/checkpoint_mnist-75sp_065802_epoch30_seed0000111.pth.tar) | |


The scripts to train the models must be run from the main directory, e.g.: ```./scripts/mnist_75sp.sh```

Examples of evaluating our trained models can be found in notebooks: [MNIST_eval_models](notebooks/MNIST_eval_models.ipynb) and
[TRIANGLES_eval_models](notebooks/TRIANGLES_eval_models.ipynb).


## Other examples of training models

To tune hyperparameters on the validation set for COLORS, TRIANGLES and MNIST, use the ```--validation``` flag.

For COLLAB, PROTEINS and D&D tuning of hyperparameters is included in the training script.

Example of running 10 weakly-supervised experiments on PROTEINS with cross-validation of hyperparameters:

```for i in $(seq 1 1 10); do seed=$(( ( RANDOM % 10000 )  + 1 )); python main.py --seed $seed -D TU --n_nodes 25 --epochs 50 --lr_decay_step 25,35,45 --test_batch_size 100 -f 64,64,64 -K 3 --aggregation mean --n_hidden 0 --readout max --dropout 0.1 --pool attn_sup_threshold_skip_skip_0 --pool_arch fc_prev --results ./checkpoints --data_dir ./data/PROTEINS | tee logs/proteins_wsup_seed"$seed".log; done```
>>>>>>> 600787e0999f2764f486a1a8c43c9ccea21b2d19


# Reference

Please cite our paper if you use our data or code.

```
@inproceedings{knyazev2019understanding,
  author = {Boris Knyazev and Graham Taylor and Mohamed Amer},
<<<<<<< HEAD
  title = {Understanding Attention in Graph Neural Networks},
=======
  title = {Understanding Attention and Generalization in Graph Neural Networks},
>>>>>>> 600787e0999f2764f486a1a8c43c9ccea21b2d19
  booktitle = {International Conference on Learning Representations (ICLR) Workshop on Representation Learning on Graphs and Manifolds},
  year = 2019,
  pdf = {http://arxiv.org/abs/1905.02850}
}
```
