import os
import sys
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')

from models.cvae import CVAE
from datasets import BodyMapDataset
from torch.utils.data import DataLoader


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--data_size', type=int, default=-1)
    parser.add_argument('--result_dir', default='result')
    parser.add_argument('--pkl', default='')
    parser.add_argument('--save_dir', default='checkpoint')
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--y_dim', type=int, default=1000)
    parser.add_argument('--pix_dim', type=int, default=256)
    parser.add_argument('--cls_num', type=int, default=12)
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--gan_type', type=str, default='VAE')
    parser.add_argument('--worker', type=int, default=10)
    parser.add_argument('--lrG', type=float, default=1e-3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--testmode', action='store_true')

    args = parser.parse_args()

    # Select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    import torch
    torch.backends.cudnn.benchmark = True
    
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn')

    args.device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

    # Make output direcotiry if not exists
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    model = CVAE(args)
    
    if args.testmode:
        # model.test('ONEHOT')
        model.test('MULTIHOT')
    else:
        model.train()

if __name__ == '__main__':
    main()
