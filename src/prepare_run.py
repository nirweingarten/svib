import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from helper import prepare_run
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-class", type=str, default="imdb", help="Kind of dataset to prepare: imagenet or imdb")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use, defaults to cpu")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    if args.data_class.lower() == 'imdb':
        print('Preparing run for IMDB...')
        prepare_run('imdb', device)
        print('success')
    elif args.data_class.lower() == 'imagenet':
        print('Preparing run for imagenet')
        prepare_run('imagenet', device)
        print('success')
    else:
        raise NotImplementedError
