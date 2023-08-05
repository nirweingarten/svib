import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPSILON = torch.tensor(1e-40).to(DEVICE)
NP_EPSILON = 1e-300
DATASET_DIR = '/D/datasets/imagenet/'
TEXTUAL_DATASETS = ('yelp', 'imdb', 'ag_news', 'cola', 'mnli')  # mr movie review (rotten tomatoes), rte
NUM_WORKERS = 1
IMAGENET_LOGITS_TRAIN_DATALOADER_PATH = '/D/datasets/imagenet/logits_dataloaders/logits_train_dataloader.pkl'
IMAGENET_LOGITS_VAL_DATALOADER_PATH = '/D/datasets/imagenet/logits_dataloaders/logits_val_dataloader.pkl'
CIFAR_LOGITS_TRAIN_DATALOADER_PATH = '/D/datasets/CIFAR/logits_dataloaders/logits_train_dataloader.pkl'
CIFAR_LOGITS_TEST_DATALOADER_PATH = '/D/datasets/CIFAR/logits_dataloaders/logits_test_dataloader.pkl'
MNIST_LOGITS_TRAIN_DATALOADER_PATH = '/D/datasets/MNIST/logits_dataloaders/logits_train_dataloader.pkl'
MNIST_LOGITS_TEST_DATALOADER_PATH = '/D/datasets/MNIST/logits_dataloaders/logits_test_dataloader.pkl'
YELP_LOGITS_TRAIN_DATALOADER_PATH = '/D/datasets/yelp/logits_dataloaders/logits_train_dataloader.pkl'
YELP_LOGITS_TEST_DATALOADER_PATH = '/D/datasets/yelp/logits_dataloaders/logits_test_dataloader.pkl'
