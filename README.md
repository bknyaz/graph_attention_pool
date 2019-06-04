https://arxiv.org/abs/1905.02850

Understanding attention in graph neural networks
Boris Knyazev, Graham W. Taylor, Mohamed R. Amer

ICLR Workshop on Representation Learning on Graphs and Manifolds.

# Examples

## MNIST
```python extract_superpixels.py -s test -t 4; python extract_superpixels.py -s train -t 4;```

## CIFAR-10
```python extract_superpixels.py -D cifar10 -c 10 -n 150 -s test -t 4; python extract_superpixels.py -D cifar10 -c 10 -n 150 -s train -t 4;```

## COLORS
```for dim in 3 8 16; do python generate_data.py --dim $dim; done```

python main.py --epochs 100 --test_batch_size 100 --seed 2038 -K 1 --aggregation sum

## TRIANGLES

python generate_data.py -D triangles --N_train 30000 --N_val 5000 --N_test 5000 --label_min 1 --label_max 10 --N_max 100 --threads 2


python main.py --epochs 100 --test_batch_size 100 -K 7 --aggregation sum -D triangles --n_hidden 64 --seed 860 --lr_decay_step 85,95 --readout max -f 64,64,64 --pool topk_gt_threshold_0_skip_skip
