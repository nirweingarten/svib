from constants import DEVICE
from helper import prepare_run


if __name__ == "__main__":
    print('Preparing run for IMDB...')
    prepare_run('imdb', DEVICE)
    print('success')
    print('Preparing run for imagenet')
    prepare_run('imagenet', DEVICE)
    print('success')
