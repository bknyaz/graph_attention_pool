# Compute superpixels for MNIST/CIFAR-10 using SLIC algorithm
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic

import numpy as np
import os
import scipy
import pickle
from skimage.segmentation import slic
from torchvision import datasets
import multiprocessing as mp
import scipy.ndimage
import scipy.spatial
import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Extract SLIC superpixels from images')
    parser.add_argument('-D', '--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to the dataset')
    parser.add_argument('-o', '--out_dir', type=str, default='./data', help='path where to save superpixels')
    parser.add_argument('-s', '--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('-t', '--threads', type=int, default=0, help='number of parallel threads')
    parser.add_argument('-n', '--n_sp', type=int, default=75, help='max number of superpixels per image')
    parser.add_argument('-c', '--compactness', type=int, default=0.25, help='compactness of the SLIC algorithm '
                                                                      '(Balances color proximity and space proximity): '
                                                                      '0.25 is a good value for MNIST '
                                                                      'and 10 for color images like CIFAR-10')
    parser.add_argument('--seed', type=int, default=11, help='seed for shuffling nodes')
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args

def process_image(params):
    
    img, index, n_images, args, to_print, shuffle = params

    assert img.dtype == np.uint8, img.dtype
    img = (img / 255.).astype(np.float32)

    n_sp_extracted = args.n_sp + 1  # number of actually extracted superpixels (can be different from requested in SLIC)
    n_sp_query = args.n_sp + (20 if args.dataset == 'mnist' else 50)  # number of superpixels we ask to extract (larger to extract more superpixels - closer to the desired n_sp)
    while n_sp_extracted > args.n_sp:
        superpixels = slic(img, n_segments=n_sp_query, compactness=args.compactness, multichannel=len(img.shape) > 2)
        sp_indices = np.unique(superpixels)
        n_sp_extracted = len(sp_indices)
        n_sp_query -= 1  # reducing the number of superpixels until we get <= n superpixels

    assert n_sp_extracted <= args.n_sp and n_sp_extracted > 0, (args.split, index, n_sp_extracted, args.n_sp)
    assert n_sp_extracted == np.max(superpixels) + 1, ('superpixel indices', np.unique(superpixels))  # make sure superpixel indices are numbers from 0 to n-1

    if shuffle:
        ind = np.random.permutation(n_sp_extracted)
    else:
        ind = np.arange(n_sp_extracted)

    sp_order = sp_indices[ind].astype(np.int32)
    if len(img.shape) == 2:
        img = img[:, :, None]

    n_ch = 1 if img.shape[2] == 1 else 3

    sp_intensity, sp_coord = [], []
    for seg in sp_order:
        mask = (superpixels == seg).squeeze()
        avg_value = np.zeros(n_ch)
        for c in range(n_ch):
            avg_value[c] = np.mean(img[:, :, c][mask])
        cntr = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
        sp_intensity.append(avg_value)
        sp_coord.append(cntr)
    sp_intensity = np.array(sp_intensity, np.float32)
    sp_coord = np.array(sp_coord, np.float32)
    if to_print:
        print('image={}/{}, shape={}, min={:.2f}, max={:.2f}, n_sp={}'.format(index + 1, n_images, img.shape,
                                                                              img.min(), img.max(), sp_intensity.shape[0]))

    return sp_intensity, sp_coord, sp_order, superpixels


if __name__ == '__main__':

    dt = datetime.datetime.now()
    print('start time:', dt)

    args = parse_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    np.random.seed(args.seed)  # to make node random permutation reproducible (not tested)

    # Read image data using torchvision
    is_train = args.split.lower() == 'train'
    if args.dataset == 'mnist':
        data = datasets.MNIST(args.data_dir, train=is_train, download=True)
        assert args.compactness < 10, ('high compactness can result in bad superpixels on MNIST')
        assert args.n_sp > 1 and args.n_sp < 28*28, (
            'the number of superpixels cannot exceed the total number of pixels or be too small')
    elif args.dataset == 'cifar10':
        data = datasets.CIFAR10(args.data_dir, train=is_train, download=True)
        assert args.compactness > 1, ('low compactness can result in bad superpixels on CIFAR-10')
        assert args.n_sp > 1 and args.n_sp < 32*32, (
            'the number of superpixels cannot exceed the total number of pixels or be too small')
    else:
        raise NotImplementedError('unsupported dataset: ' + args.dataset)

    images = data.train_data if is_train else data.test_data
    labels = data.train_labels if is_train else data.test_labels
    if not isinstance(images, np.ndarray):
        images = images.numpy()
    if isinstance(labels, list):
        labels = np.array(labels)
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy()

    n_images = len(labels)

    if args.threads <= 0:
        sp_data = []
        for i in range(n_images):
            sp_data.append(process_image((images[i], i, n_images, args, True, True)))
    else:
        with mp.Pool(processes=args.threads) as pool:
            sp_data  = pool.map(process_image, [(images[i], i, n_images, args, True, True) for i in range(n_images)])

    superpixels = [sp_data[i][3] for i in range(n_images)]
    sp_data = [sp_data[i][:3] for i in range(n_images)]
    with open('%s/%s_%dsp_%s.pkl' % (args.out_dir, args.dataset, args.n_sp, args.split), 'wb') as f:
        pickle.dump((labels.astype(np.int32), sp_data), f, protocol=2)
    with open('%s/%s_%dsp_%s_superpixels.pkl' % (args.out_dir, args.dataset, args.n_sp, args.split), 'wb') as f:
        pickle.dump(superpixels, f, protocol=2)

    print('done in {}'.format(datetime.datetime.now() - dt))
