import torch
from torchvision import transforms
from enum import Enum, auto


class SupportedDatasets(Enum):
    IMAGENET = auto()
    IMDB = auto()

EPSILON = torch.tensor(1e-40)
NP_EPSILON = 1e-300
DATASET_DIR = './datasets/imagenet/'
TEXTUAL_DATASETS = ('yelp', 'imdb', 'ag_news', 'cola', 'mnli')  # mr movie review (rotten tomatoes), rte
NUM_WORKERS = 1
IMAGENET_LOGITS_TRAIN_DATALOADER_PATH = './datasets/imagenet/logits_dataloaders/logits_train_dataloader.pkl'
IMAGENET_LOGITS_VAL_DATALOADER_PATH = './datasets/imagenet/logits_dataloaders/logits_val_dataloader.pkl'
CIFAR_LOGITS_TRAIN_DATALOADER_PATH = './datasets/CIFAR/logits_dataloaders/logits_train_dataloader.pkl'
CIFAR_LOGITS_TEST_DATALOADER_PATH = './datasets/CIFAR/logits_dataloaders/logits_test_dataloader.pkl'
MNIST_LOGITS_TRAIN_DATALOADER_PATH = './datasets/MNIST/logits_dataloaders/logits_train_dataloader.pkl'
MNIST_LOGITS_TEST_DATALOADER_PATH = './datasets/MNIST/logits_dataloaders/logits_test_dataloader.pkl'
YELP_LOGITS_TRAIN_DATALOADER_PATH = './datasets/yelp/logits_dataloaders/logits_train_dataloader.pkl'
YELP_LOGITS_TEST_DATALOADER_PATH = './datasets/yelp/logits_dataloaders/logits_test_dataloader.pkl'

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
