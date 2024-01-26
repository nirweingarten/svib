import torch
from torchvision import transforms
from enum import Enum, auto


class SupportedDatasets(Enum):
    IMAGENET = auto()
    IMDB = auto()

EPSILON = torch.tensor(1e-40)
NP_EPSILON = 1e-300
# DATASET_DIR = './datasets/imagenet/'
DATASET_DIR = '/D/datasets/imagenet/'
TEXTUAL_DATASETS = ('imdb')
NUM_WORKERS = 1
IMAGENET_LOGITS_TRAIN_DATALOADER_PATH = './datasets/imagenet/logits_dataloaders/logits_train_dataloader.pkl'
IMAGENET_LOGITS_VAL_DATALOADER_PATH = './datasets/imagenet/logits_dataloaders/logits_test_dataloader.pkl'
# IMAGENET_LOGITS_TRAIN_DATALOADER_PATH = '/D/datasets/imagenet/logits_dataloaders/logits_big_train_dataloader.pkl'
# IMAGENET_LOGITS_VAL_DATALOADER_PATH = '/D/datasets/imagenet/logits_dataloaders/logits_big_val_dataloader.pkl'
LR = 1e-4


IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Computed from the existing dataset
MAX_ENT_IMAGENET = torch.tensor(5.8972)  # Almost uiform disctribution of many possible outcomes. Empirically computed from the dataset
MAX_ENT_IMDB = torch.tensor(0.6931)  # Uniform disctribution of 2 possible outcomes. Empirically computed from the dataset
