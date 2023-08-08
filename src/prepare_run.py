from constants import DEVICE
from helper import prepare_run
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-class", type=str, default="imdb", help="Kind of dataset to prepare: imagenet or imdb")


if __name__ == "__main__":
    args = parse_args()
    if args.data_class.lower() == 'imdb':
        print('Preparing run for IMDB...')
        prepare_run('imdb', DEVICE)
        print('success')
    elif args.data_class.lower() == 'imagenet':
        print('Preparing run for imagenet')
        prepare_run('imagenet', DEVICE)
        print('success')
    else:
        raise NotImplementedError
