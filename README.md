# Examples

## MNIST
```python extract_superpixels.py -s test -t 4; python extract_superpixels.py -s train -t 4;```

## CIFAR-10
```python extract_superpixels.py -D cifar10 -c 10 -n 150 -s test -t 4; python extract_superpixels.py -D cifar10 -c 10 -n 150 -s train -t 4;```

## COLORS
```for dim in 3 8 16; do python generate_data.py --dim $dim; done```

## TRIANGLES
