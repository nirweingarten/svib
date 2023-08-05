from datasets import load_from_disk, load_dataset
from datetime import datetime
import random
from distutils.util import strtobool
import argparse
import torch
import torchvision
import numpy as np
from torchvision.datasets import CIFAR100
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST
import datasets
import numpy as np
import torch
import math
import umap.umap_ as umap
import os
import pickle
import pickle5
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.multinomial import Multinomial
from torchvision.datasets import CIFAR100, ImageNet
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from tqdm import tqdm
import gc
import cw
from torch.utils.data import Dataset
from transformer_vib import TransformerVIB, TransformerHybridModel, tokenizer, encode, text_attacks
import wandb
import time
from torch.utils.tensorboard import SummaryWriter
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the root directory where the ImageNet dataset is stored
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


# transform = transforms.Compose([transforms.Resize((299, 299)),
#                                 transforms.ToTensor()])

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ImageNet dataset and select the first 100,000 samples
train_dataset = datasets.ImageNet(root=DATASET_DIR, split='train', transform=transform)
train_subset_dataset = torch.utils.data.Subset(train_dataset, range(100000))

val_dataset = datasets.ImageNet(root=DATASET_DIR, split='val', transform=transform)
val_subset_dataset = torch.utils.data.Subset(val_dataset, range(20000))

# Create a data loader to iterate over the subset dataset
# train_data_loader = torch.utils.data.DataLoader(train_subset_dataset, batch_size=32, shuffle=True)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_data_loader = torch.utils.data.DataLoader(val_subset_dataset, batch_size=32, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

# TEST_DATASET_DIR = '/D/datasets/imagenet/test'
# test_dataset = ImageFolder(TEST_DATASET_DIR, transform=transform)
# test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# for x, y in test_data_loader:
#     break
# plt.imshow(np.transpose(x[0], (1, 2, 0))
# print(y[0])


def test_model(model, test_data_loader):
    model.eval()
    total_correct = 0
    total_incorrect = 0
    with torch.no_grad():
        for x, y in tqdm(test_data_loader):
            output = model(x.to(device))
            if type(output) == tuple:
                logits = output[-1].squeeze(0)
            else:
                logits = output
            predictions = torch.argmax(torch.softmax(logits, dim=-1), dim=1)#.to(torch.device('cpu'))
            correct_classifications = sum(predictions == y.to(device))
            incorrect_classifications = len(x) - correct_classifications
            total_correct += correct_classifications
            total_incorrect += incorrect_classifications
    model.train()
    acc = (total_correct / (total_correct + total_incorrect)).item()
    print(f"acc: {acc}")
    return acc


# test_model(model, val_data_loader)

class LogitsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
    
def get_logits_dataloader(model, original_loader, batch_size=32, whiten=False):
    logits_data_list = []
    logits_labels_list = []
    with torch.no_grad():
        for x, y in tqdm(original_loader):
            logits = model(x.to(device))
            logits_data_list.append(logits.to(torch.device('cpu')))
            logits_labels_list.append(y.to(torch.device('cpu')))

    logits_data_set = LogitsDataset(torch.concat(logits_data_list), torch.concat(logits_labels_list))
    logits_dataloader = DataLoader(logits_data_set, batch_size=batch_size, shuffle=True)
    if whiten:
        # Apply whitening to the features
        scaler = StandardScaler()
        outputs = scaler.fit_transform(torch.concat(logits_data_list).numpy())
        pca = PCA()
        whitened = pca.fit_transform(outputs)

        whitened_logits_data_set = LogitsDataset(whitened, torch.concat(logits_labels_list))
        whitened_logits_dataloader = DataLoader(whitened_logits_data_set, batch_size=batch_size, shuffle=True)
        return whitened_logits_dataloader
    else:
        return logits_dataloader
    
EPSILON = torch.tensor(1e-40).to(device)
NP_EPSILON = 1e-300

# def get_total_correlation(sigma):
#     """
#     Computes the total correlation (AKA multivariate mutual information) of a multivariate gaussian.
#     This is per https://www.mdpi.com/1099-4300/21/10/921
#     $TC(\mu,\Sigma)=\frac{1}{2}\sum_{i=1}^{D}log\left(\sigma_{ii}^{2}\right)-\frac{1}{2}log\left(\Big|\Sigma\Big|\right)$
#     """

def get_multivariate_gaussian_entropy(std, epsilon=EPSILON, num_logs=64):
    """
    $$H(N_D(\mu,\Sigma))=\frac{D}{2}(1+log(2\pi))+\frac{1}{2}log|\Sigma|$$
    Assuming a diagonal cov matrix represented as a vector
    """
    std = std.to(torch.float64)
    if not (std > 0).all():
        raise ValueError('Got a non-positive entry in diagonal cov matrix')
    D = std.shape[-1]
    if D % num_logs:
        raise ValueError('Std dim does not divide by number of log terms')
    # The log term is split into num_logs terms to avoid float precision issues (can't cast to numpy because of gradient)
    log_term = 0
    step_size = D // num_logs
    for i in range(D, 0, -step_size):
        log_term += torch.log(torch.clamp(torch.prod(std[:, i - step_size:i], dim=-1), min=epsilon))
    return torch.maximum(epsilon, ((D / 2) * (1 + np.log(2 * np.pi)) + 0.5 * log_term)).mean()


def get_multinomial_entropy(logits, epsilon=EPSILON):
    """
    Receives unactivates logits
    epsilon replaces 0 probability that results from torch's low float resolution
    """
    logits = logits.squeeze(0).to(torch.float64)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.clamp(probs, epsilon, 1 - epsilon)
    return (probs * torch.log(1/probs)).sum(dim=-1).mean()


def get_kld_between_multivariate_gaussians(mu1, std1, mu2, std2, epsilon=NP_EPSILON):
    """
    Computes batch wise KLD - Will return a tensor in the shape of batch_size where each entry is the sum over all dimensions of the KLD between the two corresponding mu and sigma.
    assuming diagonal cov matrix as 1d ndarray

    To test this use:
    from torch.distributions import kl
    base_dist1 = MultivariateNormal(base_mu[0], torch.eye(base_std.shape[-1]).to(device)*base_std[0])
    base_dist2 = MultivariateNormal(base_mu[1], torch.eye(base_std.shape[-1]).to(device)*base_std[1])
    new_dist1 = MultivariateNormal(mu[0], torch.eye(base_std.shape[-1]).to(device)*std[0])
    new_dist2 = MultivariateNormal(mu[1], torch.eye(base_std.shape[-1]).to(device)*std[1])
    kl1 = kl.kl_divergence(base_dist1, new_dist1)
    kl2 = kl.kl_divergence(base_dist2, new_dist2)
    (kl1 + kl2) / 2
    get_kld_between_multivariate_gaussians(base_mu[0].unsqueeze(0), base_std[0].unsqueeze(0), mu[0].unsqueeze(0), std[0].unsqueeze(0))
    get_kld_between_multivariate_gaussians(base_mu[1].unsqueeze(0), base_std[1].unsqueeze(0), mu[1].unsqueeze(0), std[1].unsqueeze(0))
    get_kld_between_multivariate_gaussians(base_mu[0:2], base_std[0:2], mu[0:2], std[0:2])
    """
    MINIMAL_LOG_VALUE = -1500

    mu1 = mu1.cpu().numpy().astype(np.float128)
    std1 = std1.cpu().numpy().astype(np.float128)
    mu2 = mu2.cpu().numpy().astype(np.float128)
    std2 = std2.cpu().numpy().astype(np.float128)

    N, D = mu1.shape

    # Compute the determinante log ratio term
    log_term = np.log(np.prod(std2, axis=-1) / np.prod(std1, axis=-1))

    # In case one of the denominator values has zeroized due to floating point percision
    log_term = np.where(log_term == -np.inf, MINIMAL_LOG_VALUE, log_term)

    # Compute the trace term
    trace_term = ((1 / std2) * std1).sum(axis=-1)

    # Compute the quadratic term
    mu_diff = mu2 - mu1
    quadratic_term = np.sum((mu_diff * (1 / std2) * mu_diff), axis=-1)

    # Compute the KLD for each pair of Gaussians
    kld = 0.5 * (log_term - D + trace_term + quadratic_term)

    return kld

def reparametrize(mu, std, device):
    """
    Performs reparameterization trick z = mu + epsilon * std
    Where epsilon~N(0,1)
    """
    mu = mu.expand(1, *mu.size())
    std = std.expand(1, *std.size())
    eps = torch.normal(0, 1, size=std.size()).to(device)
    return mu + eps * std

class VIB(nn.Module):
    """
    Classifier with stochastic layer and KL regularization
    """
    def __init__(self, hidden_size, output_size, device):
        super(VIB, self).__init__()
        self.device = device
        self.description = 'Vanilla IB VAE as per the paper'
        self.hidden_size = hidden_size
        self.k = hidden_size // 2
        self.output_size = output_size
        self.train_loss = []
        self.test_loss = []

        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(self.k, output_size)

        # Xavier initialization
        for _, module in self._modules.items():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                        module.bias.data.zero_()
                        continue
            for layer in module:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                            layer.bias.data.zero_()
        
        # Init with a small bias in std to create initial high entropy
        # biased_layer = self.encoder[2]
        # slice_size = biased_layer.bias.shape[0] // 2
        # new_bias = torch.concat((biased_layer.bias[:slice_size], torch.ones(slice_size) * 0.1))
        # biased_layer.bias.data = new_bias


    def forward(self, x):
        z_params = self.encoder(x)
        mu = z_params[:, :self.k]
        # softplus transformation (soft relu) and a -5 bias is added as in the paper
        # std = F.softplus(z_params[:, self.k:] - 5, beta=1)
        std = F.softplus(z_params[:, self.k:] - 1, beta=1)
        # std = z_params[:, self.k:]
        # std = F.softplus(z_params[:, self.k:], beta=1)
        if self.training:
            z = reparametrize(mu, std, self.device)
        else:
            z = mu.clone().unsqueeze(0)
        n = Normal(mu, std)
        log_probs = n.log_prob(z.squeeze(0))  # These may be positive as this is a PDF
        
        logits = self.classifier(z)
        return (mu, std), log_probs, logits
   
def vib_loss(logits, labels, mu, std, beta):
    classification_loss = nn.CrossEntropyLoss()(logits.squeeze(0), labels)  # In torch cross entropy function applies the softmax itself
    normalization_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum()
    # KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return classification_loss + beta * normalization_loss


def fgsm_attack(data, epsilon, data_grad, is_targeted=False, is_image=True):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    if is_targeted:
        sign_data_grad *= -1
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_data = data + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    if is_image:
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
    # Return the perturbed image
    return perturbed_data


def run_adverserial_attacks(model, device, test_loader, epsilon, target_label=None, is_image=True, print_results=False, attack_type='fgs', mean=(), std=()):

    # Accuracy counter
    correct = 0
    relevant_pertrubations = 0
    total_succesful_attacks = 0
    l2_dist_for_sx_attack_list = []
    adv_examples = []

    model.eval()

    if isinstance(model, HybridModel):
        model.set_return_only_logits(True)
    if isinstance(test_loader.dataset, torch.utils.data.dataset.Subset):
        classes = test_loader.dataset.dataset.classes
    else:
        classes = test_loader.dataset.classes

    # Loop over all examples in test set
    for data, labels in tqdm(test_loader):
        # Send the data and label to the device
        data, labels = data.to(device), labels.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        if type(output) == tuple:
            _, _, output = model(data)

        # get the index of the max log-probability
        init_pred = output.squeeze(0).max(dim=-1, keepdim=True)[1]

        hit_vector = init_pred.squeeze(1) == labels
#         relevant_targeted_attack_vector = init_pred.squeeze(1) != target_label
        relevant_targeted_attack_vector = labels != target_label
        if target_label:
            relevant_attack_idx = torch.nonzero(relevant_targeted_attack_vector & (
                init_pred.squeeze(1) != target_label)).flatten()
        else:
            relevant_attack_idx = torch.nonzero(hit_vector).flatten()

        # If the initial prediction is wrong, don't bother attacking, just move on
        if len(relevant_attack_idx) == 0:
            correct += hit_vector.sum().to(torch.device('cpu'))
            continue

        relevant_output = output.squeeze(0)[relevant_attack_idx]
        relevant_labels = labels[relevant_attack_idx]

        if attack_type.lower() == 'fgs':
            # Calculate the loss for the gradient
            if target_label:
                # Targeted attack
                #             loss = F.nll_loss(relevant_output, torch.tensor([target_label]).to(device))
                loss = F.nll_loss(relevant_output, torch.tensor(
                    target_label).expand(relevant_output.shape[0]).to(device))
            else:
                # Untargeted attack
                loss = F.nll_loss(relevant_output, relevant_labels)
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect ``datagrad``
            data_grad = data.grad[relevant_attack_idx].data
            # Call FGSM Attack
            perturbed_data = fgsm_attack(data[relevant_attack_idx], epsilon, data_grad, is_targeted=bool(
                target_label), is_image=is_image)
        elif attack_type.lower() == 'cw':  # Carlini Wagner L2 attack
            data = data.detach()
            inputs_box = (min((0 - m) / s for m, s in zip(mean, std)),
                          max((1 - m) / s for m, s in zip(mean, std)))
            adversary = cw.L2Adversary(targeted=bool(target_label),
                                    confidence=0.0,
                                    search_steps=10,
                                    box=inputs_box,
                                    optimizer_lr=5e-4,
                                    num_classes=len(classes),
                                    device=device)
            # a batch of any attack targets
            if bool(target_label):
                attack_targets = torch.ones(data.size(0)).to(torch.int64) * target_label
                attack_targets = attack_targets.to(device)
            else:
                attack_targets = labels
            perturbed_data = adversary(model, data[relevant_attack_idx], attack_targets[relevant_attack_idx], to_numpy=False)
            # assert isinstance(perturbed_data, torch.FloatTensor)
            # assert perturbed_data.size() == data.size()
            perturbed_data = perturbed_data.to(device)

        # Re-classify the perturbed image
        perturbed_output = model(perturbed_data)
        if type(perturbed_output) == tuple:
            perturbed_output = perturbed_output[-1]

        relevant_pertrubations += perturbed_output.squeeze(0).shape[0]

        # Check for success
        perturbed_pred = perturbed_output.squeeze(0).max(
            dim=-1, keepdim=True)[1]  # get the index of the max log-probability

        correct_perturbed_classifications = perturbed_pred.flatten(
        ) == labels[relevant_attack_idx]


        # Unsuccessful attack
        if target_label:
            unsuccessful_attack_vector = perturbed_pred.flatten() != target_label
        else:
            unsuccessful_attack_vector = correct_perturbed_classifications

        num_unsuccessful_attacks = unsuccessful_attack_vector.sum()
        successful_attack_idx = torch.nonzero(
            ~unsuccessful_attack_vector.flatten())
        num_successful_attacks = (~unsuccessful_attack_vector).sum()
        unsuccessful_attack_idx = torch.nonzero(
            unsuccessful_attack_vector.flatten())

        total_succesful_attacks += num_successful_attacks
        if len(successful_attack_idx):
            l2_dist_for_sx_attack_list.append(torch.dist(perturbed_data, data[successful_attack_idx], p=2).item() / len(successful_attack_idx))

        # Diff between unsuccessful attack and correct classification
        # In correct classification we also take into account unrelevant items and those who didn't reach the required target
#         correct_batch_classifications = (~relevant_targeted_attack_vector & hit_vector).sum() + correct_perturbed_classifications.sum()
        correct_batch_classifications = correct_perturbed_classifications.sum()

        correct += correct_batch_classifications.to(torch.device('cpu'))
        # Special case for saving 0 epsilon examples
        if (epsilon == 0) and (len(adv_examples) < 5):
            adv_ex = perturbed_data[unsuccessful_attack_idx][0].squeeze(
            ).detach().cpu().numpy()
            adv_examples.append((init_pred[unsuccessful_attack_idx][0].item(
            ), perturbed_pred[unsuccessful_attack_idx][0].item(), adv_ex))

        if num_successful_attacks and (epsilon != 0):
            # Save some adv examples for visualization later
            #             if (len(adv_examples) < 5) and (not bool(target_label) or (target_label != labels.item())):
            if (len(adv_examples) < 5):
                adv_ex = perturbed_data[successful_attack_idx][0].squeeze(
                ).detach().cpu().numpy()
                adv_examples.append((init_pred[successful_attack_idx][0].item(
                ), perturbed_pred[successful_attack_idx][0].item(), adv_ex))

    # Calculate final accuracy for this epsilon
    # float(len(test_loader) * test_loader.batch_size)
    final_acc = correct / relevant_pertrubations
    # float(len(test_loader) * test_loader.batch_size)
    succesful_attack_rate = total_succesful_attacks.item() / relevant_pertrubations
    if print_results:
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\t %succesful attacks: {}\t Out of total of {} data points".format(epsilon,
            correct, relevant_pertrubations, final_acc, succesful_attack_rate, len(test_loader) * test_loader.batch_size))

    if isinstance(model, HybridModel):
        model.set_return_only_logits(False)

    # Return the accuracy and an adversarial example
    return final_acc, succesful_attack_rate, adv_examples, np.mean(l2_dist_for_sx_attack_list)


def attack_and_eval(model, device, test_data_loader, target_label, epsilons, mean=(), std=()):
    print(f'### Running adverserial attacks ###')
    untargeted_accuracies = []
    untargeted_examples = []
    untargeted_total_succesful_attacks_list = []
    
    targeted_accuracies = []
    targeted_examples = []
    targeted_total_succesful_attacks_list = []
    
    # CW targeted attack. Using a subset otherwise this takes forever
    test_subset_dataset = torch.utils.data.Subset(test_data_loader.dataset, range(len(test_data_loader.dataset) // 100))  # Using first 1% of data for targeted attacks
    test_subset_dataloader = DataLoader(test_subset_dataset, batch_size=32, shuffle=False)  # Not shuffeling as we go over only a subset
    acc, total_succesful_attacks, ex, avg_l2_dist_for_sx_attack = run_adverserial_attacks(model, device, test_subset_dataloader, epsilon=1, is_image=False,
                                                                                                                    target_label=target_label, attack_type='cw', mean=mean, std=std)
    targeted_accuracies.append(acc)
    targeted_total_succesful_attacks_list.append(total_succesful_attacks)    
    targeted_examples.append(ex)

    # FGS untargeted attack
    for eps in epsilons:
        acc, total_succesful_attacks, ex, _ = run_adverserial_attacks(model, device, test_data_loader, eps, is_image=False)
        untargeted_accuracies.append(acc)
        untargeted_total_succesful_attacks_list.append(total_succesful_attacks)    
        untargeted_examples.append(ex)


    return untargeted_accuracies, untargeted_examples, untargeted_total_succesful_attacks_list, targeted_accuracies, targeted_examples, targeted_total_succesful_attacks_list, avg_l2_dist_for_sx_attack


def loop_data(model, train_dataloader, test_dataloader, beta, writer, epochs,
              device, optimizer=None, scheduler=None, eta=0.001,
              num_minibatches=1,  loss_type='vib',
              clip_grad=False, clip_loss=False, kl_rate_loss=False, max_grad_norm=2):

    model.train()

    epoch_h_z_x_array = np.zeros(epochs)
    epoch_h_z_y_array = np.zeros(epochs)

    for e in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_classification_loss = 0
        epoch_total_kld = 0
        epoch_ratio1 = 0
        epoch_ratio2 = 0
        epoch_ratio3 = 0
        epoch_ratio4 = 0
        epoch_mean_y_entropy = 0
        epoch_mean_z_entropy = 0
        epoch_rate_term = 0
        epoch_distortion_term = 0

        if loss_type == 'dyn_beta':
            if e < 2:
                beta = EPSILON
            else:
                with torch.no_grad():
                    epoch_i_z_x_delta = epoch_h_z_x_array[e - 1] - epoch_h_z_x_array[e - 2]
                    epoch_i_z_y_delta = epoch_h_z_y_array[e - 1] - epoch_h_z_y_array[e - 2] 
                    beta = epoch_i_z_y_delta / epoch_i_z_x_delta
                    if (epoch_i_z_x_delta == 0) or (beta > 1):
                        beta = EPSILON

        for batch_num, (embeddings, labels) in enumerate(train_dataloader):
            # Compute base z distribution
            if loss_type == 'ppo':
                with torch.no_grad():
                    x = embeddings.to(device)
                    (base_mu, base_std), base_log_probs, _ = model(x)

            for i in range(num_minibatches):
                x = embeddings.to(device)
                y = labels.to(device)
                (mu, std), log_probs, logits = model(x)
#                 activated = torch.softmax(logits, -1)

                batch_h_z_x = get_multivariate_gaussian_entropy(std)
                batch_h_z_y = get_multinomial_entropy(logits)

                with torch.no_grad():
                    epoch_h_z_x_array[e] += batch_h_z_x.cpu().detach() / len(train_dataloader)
                    epoch_h_z_y_array[e] += batch_h_z_y.cpu().detach() / len(train_dataloader)

                kld_from_std_normal = (-0.5 * (1 + 2 * std.log() -
                                    mu.pow(2) - std.pow(2))).sum(1).mean(0, True)

                classification_loss = nn.CrossEntropyLoss()(logits.squeeze(0), y)

                if kl_rate_loss:
                    rate_term = kld_from_std_normal.sum() - batch_h_z_x
                else:
                    rate_term = get_multivariate_gaussian_entropy(torch.ones(std.shape[-1]).unsqueeze(0).to(device)) - batch_h_z_x
                # distortion_term = classification_loss + batch_h_z_y
                distortion_term = classification_loss - batch_h_z_y
                epoch_rate_term += rate_term.item() / len(train_dataloader)
                epoch_distortion_term += distortion_term.item() / len(train_dataloader)

                if loss_type == 'ppo':
                    if i == 0:
                        kld_from_base_dist = 0
                        ratio = 0
                    else:
                        with torch.no_grad():
                            kld_from_base_dist = get_kld_between_multivariate_gaussians(base_mu, base_std, mu, std).mean()
                    with torch.no_grad():
                        ratio = kld_from_base_dist / \
                            (kld_from_std_normal + kld_from_base_dist)
                    minibatch_loss = classification_loss + \
                        (beta * ratio).mean() * kld_from_std_normal.sum()
                elif loss_type == 'vub':
                    # minibatch_loss = rate_term + beta * distortion_term
                    if clip_loss:
                        max_regularization_value = (abs(beta) * classification_loss).item()
                        min_regularization_value = torch.tensor(0).to(device)
                        # minibatch_loss = torch.clamp(rate_term, min=torch.tensor(0).to(device), max=(beta * distortion_term).item()) + beta * distortion_term  #  THIS WORKED MORE OR LESS LIKE REGULAR VIB (- H(Y|Z))
                        # minibatch_loss = torch.clamp(rate_term, min=torch.tensor(0).to(device), max=(beta * distortion_term).item()) + classification_loss + beta * batch_h_z_y  #  DIDN'T TRY THIS YET
                        minibatch_loss = torch.clamp(rate_term, min=min_regularization_value, max=(max_regularization_value)) + classification_loss - torch.clamp(beta * batch_h_z_y, min=min_regularization_value, max=max_regularization_value)
                        # minibatch_loss = - torch.clamp(batch_h_z_x, min=torch.tensor(0).to(device), max=classification_loss) + classification_loss + beta * torch.clamp(batch_h_z_y, min=torch.tensor(0).to(device), max=(classification_loss).item())    THIS DIDNT WORK
                        # minibatch_loss = - torch.clamp(batch_h_z_x, min=torch.tensor(0).to(device), max=(classification_loss/2).item()) \
                        #     - beta * torch.clamp(batch_h_z_y, min=torch.tensor(0).to(device), max=(classification_loss/2).item()) \
                        #         + beta * classification_loss
                    else:
                        # minibatch_loss = rate_term + beta * distortion_term
                        minibatch_loss = - batch_h_z_x + beta * (classification_loss - batch_h_z_y)
                elif loss_type == 'vib':
                    minibatch_loss = classification_loss + beta * kld_from_std_normal
                elif loss_type == 'pereyara':
                    minibatch_loss = classification_loss - beta * batch_h_z_y
                else:
                    raise NotImplementedError

                optimizer.zero_grad()
                minibatch_loss.backward()
                if clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    epoch_total_kld += kld_from_std_normal / num_minibatches
                    epoch_classification_loss += classification_loss.item() / num_minibatches
                    if loss_type == 'ppo':
                        if i == 0:
                            epoch_ratio1 += ratio
                        elif i == 1:
                            epoch_ratio2 += ratio
                        elif i == 2:
                            epoch_ratio3 += ratio
                        elif i == 3:
                            epoch_ratio4 += ratio

            epoch_loss += minibatch_loss.item()

        epoch_loss /= batch_num
        model.train_loss.append(epoch_loss)
        # writer.add_scalar("charts/epoch_train_loss", epoch_loss, e)
        writer.add_scalar(
            "charts/epoch_classification_loss", epoch_classification_loss / len(train_dataloader), e)
        writer.add_scalar(
            "charts/epoch_total_kld", epoch_total_kld / len(train_dataloader), e)

        # Entropy terms cancel out, these deltas are percise
        if e > 0:
            epoch_i_z_x_delta = epoch_h_z_x_array[e] - epoch_h_z_x_array[e - 1]
            epoch_i_z_y_delta = epoch_h_z_y_array[e] - epoch_h_z_y_array[e - 1] 
            # writer.add_scalar("charts/epoch_i_z_x_delta", epoch_i_z_x_delta, e)
            # writer.add_scalar("charts/epoch_i_z_y_delta", epoch_i_z_y_delta, e)

        # if loss_type in ('dyn_beta', 'vub'):
            # h_z_x and h_z_y are the variable negative terms in I(Z;X) and I(Z;Y) and hence are in reverse ratio to I()
        writer.add_scalar("charts/epoch_h_z_x", epoch_h_z_x_array[e], e)
        writer.add_scalar("charts/epoch_h_z_y", epoch_h_z_y_array[e], e)
        
        # if loss_type == 'vub':
        writer.add_scalar("charts/epoch_rate_term", epoch_rate_term, e)
        writer.add_scalar("charts/epoch_distortion_term", epoch_distortion_term, e)

        if loss_type == 'dyn_beta':
            writer.add_scalar("charts/epoch_dyn_beta", beta, e)

        if loss_type == 'ppo':
            writer.add_scalar(
                "charts/epoch_ratio1", epoch_ratio1 / len(train_dataloader), e)
            writer.add_scalar(
                "charts/epoch_ratio2", epoch_ratio2 / len(train_dataloader), e)
            writer.add_scalar(
                "charts/epoch_ratio3", epoch_ratio3 / len(train_dataloader), e)
            writer.add_scalar(
                "charts/epoch_ratio4", epoch_ratio4 / len(train_dataloader), e)
            writer.add_scalar(
                "charts/avg_epoch_ratio", (epoch_ratio1 + epoch_ratio2 +
                                        epoch_ratio3 + epoch_ratio4) / (len(train_dataloader) * 4), e)
            writer.add_scalar(
                "charts/total_epoch_ratio", (epoch_ratio1 + epoch_ratio2 + epoch_ratio3 + epoch_ratio4) / len(train_dataloader), e)

        if (not ((e + 1) % 10)) or (e == 0):
            # test loss
            model.eval()
            total_correct = 0
            total_incorrect = 0
            epoch_val_classification_loss = 0
            with torch.no_grad():
                for batch_num, (embeddings, labels) in enumerate(test_dataloader):
                    x = embeddings.to(device)
                    y = labels.to(device)
                    (mu, std), log_probs, logits = model(x)
                    logits = logits.squeeze(0)
                    epoch_val_classification_loss += nn.CrossEntropyLoss()(logits, y) / len(test_dataloader)
                    # epoch_test_loss += vib_loss(logits, y, mu, std, beta).item()
                    # TODO: add this line
                    # epoch_test_loss += vib_loss(logits, y, mu, std, beta).item() / len(train_dataloader)
                    predictions = torch.argmax(torch.softmax(logits, dim=-1), dim=1)
                    correct_classifications = sum(predictions == labels.to(device))
                    incorrect_classifications = len(x) - correct_classifications
                    total_correct += correct_classifications
                    total_incorrect += incorrect_classifications
                acc = (total_correct / (total_correct + total_incorrect)).item()

            model.test_loss.append(epoch_val_classification_loss.item())
            writer.add_scalar("charts/epoch_val_classification_loss", epoch_val_classification_loss, e)
            writer.add_scalar("charts/epoch_val_accuracy", acc, e)
            model.train()


class HybridModel(nn.Module):
    """
    Head is a pretrained model, classifier is VIB
    fc_name should be 'fc2' for inception-v3 (imagenet) and mnist-cnn, '_fc' for efficient-net (CIFAR)
    """
    def __init__(self, base_model, vib_model, device, fc_name, return_only_logits=False):
        super(HybridModel, self).__init__()
        self.device = device
        self.base_model = base_model
        setattr(self.base_model, fc_name, torch.nn.Identity())
        self.vib_model = vib_model
        self.train_loss = []
        self.test_loss = []
        self.freeze_base()
        self.return_only_logits = return_only_logits

    def set_return_only_logits(self, bool_value):
        self.return_only_logits = bool_value
    
    def freeze_base(self):
        # Freeze the weights of the inception_model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        # Freeze the weights of the inception_model
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        encoded = self.base_model(x)
        (mu, std), log_probs, logits = self.vib_model(encoded)
        if self.return_only_logits:
            return logits.squeeze(0)
        else:
            return ((mu, std), log_probs, logits)
    

def get_dataloaders(data_class, logits=False):
    
    if data_class == 'mnist':
        if logits:
            with open(MNIST_LOGITS_TRAIN_DATALOADER_PATH, 'rb') as f:
                train_data_loader = pickle5.load(f)
            with open(MNIST_LOGITS_TEST_DATALOADER_PATH, 'rb') as f:
                val_data_loader = pickle5.load(f)
            return train_data_loader, val_data_loader
        else:
            dataset = MNIST
            dataset_dir = '/D/datasets/MNIST'
            batch_size = 128
            train_transform = transforms.Compose(
                [transforms.Resize((28, 28)), transforms.ToTensor()])
            test_transform = train_transform
            train_kwargs = {'train': True}
            test_kwargs = {'train': False}

    elif data_class == 'cifar':
        if logits:
            with open(CIFAR_LOGITS_TRAIN_DATALOADER_PATH, 'rb') as f:
                train_data_loader = pickle5.load(f)
            with open(CIFAR_LOGITS_TEST_DATALOADER_PATH, 'rb') as f:
                val_data_loader = pickle5.load(f)
            return train_data_loader, val_data_loader
        else:
            dataset = CIFAR100
            dataset_dir = '/D/datasets/CIFAR'
            batch_size = 32
            train_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_kwargs = {'train': True}
            test_kwargs = {'train': False}

    elif data_class == 'imagenet':
        if logits:
            with open(IMAGENET_LOGITS_TRAIN_DATALOADER_PATH, 'rb') as f:
                train_data_loader = pickle5.load(f)
            with open(IMAGENET_LOGITS_VAL_DATALOADER_PATH, 'rb') as f:
                val_data_loader = pickle5.load(f)
            return train_data_loader, val_data_loader
        else:
            dataset = ImageNet
            dataset_dir = '/D/datasets/imagenet/'
            batch_size = 32
            # TODO: Consider adding data augmentation to train transform
            train_transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            test_transform = train_transform
            train_kwargs = {'split': 'train'}
            test_kwargs = {'split': 'val'}

    elif data_class in TEXTUAL_DATASETS:
        if logits:
            logits_train_dataloader_path = f'/D/datasets/{data_class}/logits_dataloaders/logits_train_dataloader.pkl'
            logits_test_dataloader_path = f'/D/datasets/{data_class}/logits_dataloaders/logits_test_dataloader.pkl'
            with open(logits_train_dataloader_path, 'rb') as f:
                train_data_loader = pickle5.load(f)
            with open(logits_test_dataloader_path, 'rb') as f:
                val_data_loader = pickle5.load(f)
            return train_data_loader, val_data_loader
        else:
            if data_class == 'yelp':
                dataset = load_dataset('yelp_polarity')
            elif data_class == 'cola':
                dataset = load_dataset('glue', 'cola')
            else:
                dataset = load_dataset(data_class)
            dataset = dataset.map(encode, batched=True)
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

            train_dataset = dataset['train']
            test_dataset = dataset['test']

            train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=8)
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8)
            return train_dataloader, test_dataloader
    else:
        raise NotImplementedError

    if os.path.isdir(dataset_dir):
        train_data = dataset(root=dataset_dir, transform=train_transform, **train_kwargs)
        test_data = dataset(root=dataset_dir, transform=test_transform, **test_kwargs)
    else:
        os.mkdir(DATASET_DIR)
        train_data = dataset(root=dataset_dir, train=True,
                                download=True, transform=train_transform)
        test_data = dataset(root=dataset_dir, train=False,
                                download=True, transform=test_transform)

    train_loader = DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            drop_last=True)

    test_loader = DataLoader(test_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=NUM_WORKERS,
                            drop_last=False)

    classes = train_data.classes

    return train_loader, test_loader


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
#         x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7 * 4)
        x = F.relu(self.fc1(x))
#         x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_and_eval_cdlvm(data_class, betas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                         epsilons=[0.1, 0.35, 0.4, 0.45, 0.5], loss_type='vib',
                          kl_rate_loss=False, clip_grad=False, clip_loss=True,
                         num_minibatches=1, num_runs=1, num_epochs=0, text_attack_type='BAEGarg2019'):
    """
    CDLVM == conditional deep latent variational model
    """
    LR = 1e-4
    results_dict = {}
    current_time = datetime.now()
    formatted_time = current_time.strftime("%d.%m_%H:%M")
    pkl_name = data_class + f'_{loss_type}'
    if kl_rate_loss:
        pkl_name += '_kl_rate'
    if clip_grad:
        pkl_name += '_clip_grad'
    if clip_loss:
        pkl_name += '_clip_loss'
    save_path = f'/D/models/dicts/{pkl_name}_{formatted_time}.pkl'

    os.environ["WANDB_SILENT"] = "true"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if data_class == 'mnist':
        epochs = 250
        hidden_size = 256
        output_size = 10
        pretrained_path = '/D/models/pretrained/mnist_model.pkl'
        fc_name = 'fc2'
        target_label = 1
        max_grad_norm = 0.001
        transformation_mean = (0, 0, 0)
        transformation_std = (0, 0, 0)
    elif data_class == 'cifar':
        epochs = 100
        hidden_size = 1280
        output_size = 100
        pretrained_path = '/D/models/pretrained/cifar_efficientnet.pkl'
        fc_name = '_fc'
        target_label = 15  # Camel
        max_grad_norm = 2.5
        transformation_mean = (0.4914, 0.4822, 0.4465)
        transformation_std = (0.2023, 0.1994, 0.2010)
    elif data_class == 'imagenet':
        fc_name = 'fc'
        epochs = 100
        hidden_size = 2048
        output_size = 1000
        pretrained_path = '/D/models/pretrained/inceptionv3.pkl'
        target_label = 805  # soccer ball
        max_grad_norm = 5
        transformation_mean = (0.485, 0.456, 0.406)
        transformation_std = (0.229, 0.224, 0.225)
    elif data_class in ('yelp', 'imdb', 'cola'):
        fc_name = 'classifier'
        epochs = 200
        hidden_size = 768
        output_size = 2
        pretrained_path = f'/D/models/pretrained/pretrained_bert_{data_class}.pkl'
        target_label = 0  # Not relevant
        max_grad_norm = 5
    elif data_class.replace('-', '_') == 'mnli':
        fc_name = 'classifier'
        epochs = 200
        hidden_size = 768
        output_size = 3
        pretrained_path = f'/D/models/pretrained/pretrained_bert_{data_class}.pkl'
        target_label = 0  # Not relevant
        max_grad_norm = 5
    elif data_class.replace('-', '_') == 'ag_news':
        fc_name = 'classifier'
        epochs = 200
        hidden_size = 768
        output_size = 4
        pretrained_path = f'/D/models/pretrained/pretrained_bert_{data_class}.pkl'
        target_label = 0  # Not relevant
        max_grad_norm = 5
    else:
        raise NotImplementedError
    
    if num_epochs:
        print(f'### Overiding original epochs of {epochs} with user defined {num_epochs}')
        epochs = num_epochs

    logits_train_data_loader, logits_test_data_loader = get_dataloaders(data_class, logits=True)
    train_data_loader, test_data_loader = get_dataloaders(data_class, logits=False)

    if loss_type == 'vanilla':
        print(f"\n### Evaluating pretrained vanilla model ###")
        pretrained_model = torch.load(pretrained_path)
        pretrained_model.to(device)
        pretrained_model.eval()
        vanilla_run_name = f'vanilla_model_{formatted_time}'
        wandb_run = wandb.init(
            project='dynamic_beta',
            entity=None,
            sync_tensorboard=True,
            config=None,
            name=vanilla_run_name,
            monitor_gym=False,
            save_code=True,)
        writer = SummaryWriter(f"runs/{vanilla_run_name}")
        test_accuracy = test_model(pretrained_model, test_data_loader)
        untargeted_accuracies, untargeted_examples, untargeted_total_succesful_attacks_list, targeted_accuracies, targeted_examples, targeted_total_succesful_attacks_list, avg_l2_dist_for_sx_targeted_attack = attack_and_eval(pretrained_model, device, test_data_loader, target_label, epsilons, mean=transformation_mean, std=transformation_std)
        wandb_run.finish()
        results_dict['pretrained_vanilla_model'] = {
            'dict_name': pkl_name,
            'beta': 0,
            'fgs_epsilons': epsilons,
            'test_accuracy': test_accuracy,
            'untargeted_accuracies': untargeted_accuracies,
            'untargeted_total_succesful_attacks_list': untargeted_total_succesful_attacks_list,
            'untargeted_examples': untargeted_examples,
            'targeted_accuracies': targeted_accuracies,
            'targeted_total_succesful_attacks_list': targeted_total_succesful_attacks_list,
            'targeted_examples': targeted_examples,
            'avg_l2_dist_for_sx_targeted_attack': avg_l2_dist_for_sx_targeted_attack,
        }
        del(pretrained_model)
        with open(save_path, 'wb') as f:
            pickle.dump(results_dict, f)
            print(f'Saved dict to {save_path}')
        print(f'\n\
            ###### Run summary: Vanilla model ######\n\
            test acc: {test_accuracy}\n\
            untargeted succesful attacks at eps={epsilons[0]}: {untargeted_total_succesful_attacks_list[0]}\n\
            untargeted succesful attacks at eps={epsilons[-1]}: {untargeted_total_succesful_attacks_list[-1]}\n\
            untargeted acc at eps={epsilons[0]}: {untargeted_accuracies[0]}\n\
            untargeted acc at eps={epsilons[-1]}: {untargeted_accuracies[-1]}\n\
            targeted succesful attacks at eps={epsilons[0]}: {targeted_total_succesful_attacks_list[0]}\n\
            targeted succesful attacks at eps={epsilons[-1]}: {targeted_total_succesful_attacks_list[-1]}\n\
            targeted acc at eps={epsilons[0]}: {targeted_accuracies[0]}\n\
            targeted acc at eps={epsilons[-1]}: {targeted_accuracies[-1]}\n\
            avg l2 distance for succesful cw targeted attack: {avg_l2_dist_for_sx_targeted_attack}\n\
            ')
        return


    for beta in betas:
        for run_num in range(num_runs):

            run_name = f"{pkl_name}_beta_{beta}_run_{run_num}_{formatted_time}"

            wandb_run = wandb.init(
                project='dynamic_beta',
                entity=None,
                sync_tensorboard=True,
                config=None,
                name=run_name,
                monitor_gym=False,
                save_code=True,)
            writer = SummaryWriter(f"runs/{run_name}")

            print(f"\n\n### Started training {run_name} ###")

            if data_class in TEXTUAL_DATASETS:
                vib_classifier = TransformerVIB(hidden_size, output_size, device).to(device)        
            else:
                vib_classifier = VIB(hidden_size, output_size, device).to(device)        

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, vib_classifier.parameters()), LR / num_minibatches, betas=(0.5, 0.999))
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

            try:
                loop_data(vib_classifier, logits_train_data_loader, logits_test_data_loader, beta,
                        num_minibatches=num_minibatches, writer=writer, epochs=epochs, device=device,
                        optimizer=optimizer, scheduler=scheduler, loss_type=loss_type,
                        clip_grad=clip_grad, clip_loss=clip_loss, max_grad_norm=max_grad_norm,
                          kl_rate_loss=kl_rate_loss)
                print(f'### Finished training, evaluating... ###')
                
                if data_class in TEXTUAL_DATASETS:
                    with open(pretrained_path, 'rb') as f:
                        pretrained_model = pickle.load(f)
                    pretrained_model.to(device)
                    hybrid_model = TransformerHybridModel(pretrained_model, vib_classifier, device, fc_name=fc_name)
                    hybrid_model.freeze_base()
                    hybrid_model.to(device)
                    test_accuracy, deep_word_acc_under_attack, deep_word_avg_pertrubed_words_prct, deep_word_attack_success_rate, pwws_acc_under_attack, pwws_avg_pertrubed_words_prct, pwws_attack_success_rate = text_attacks(hybrid_model, data_class)
                    results_dict[run_name] = {
                        'dict_name': pkl_name,
                        'vib_classifier': vib_classifier.to('cpu'),
                        'beta': beta,
                        'test_accuracy': test_accuracy,
                        'deep_word_acc_under_attack': deep_word_acc_under_attack,
                        'deep_word_avg_pertrubed_words_prct': deep_word_avg_pertrubed_words_prct,
                        'deep_word_attack_success_rate': deep_word_attack_success_rate,
                        'pwws_acc_under_attack': pwws_acc_under_attack,
                        'pwws_avg_pertrubed_words_prct': pwws_avg_pertrubed_words_prct,
                        'pwws_attack_success_rate': pwws_attack_success_rate
                                            }
                    print(f'\n\
                        ###### Run summary: beta={beta} ######\n\
                        test acc: {test_accuracy}\n\
                        deep_word_attack_success_rate: {deep_word_attack_success_rate}\n\
                        deep_word_acc_under_attack: {deep_word_acc_under_attack}\n\
                        deep_word_avg_pertrubed_words_prct: {deep_word_avg_pertrubed_words_prct}\n\
                        pwws_attack_success_rate: {pwws_attack_success_rate}\n\
                        pwws_acc_under_attack: {pwws_acc_under_attack}\n\
                        pwws_avg_pertrubed_words_prct: {pwws_avg_pertrubed_words_prct}\n\
                        ')

                else:
                    test_accuracy = test_model(vib_classifier, logits_test_data_loader)
                    pretrained_model = torch.load(pretrained_path)
                    pretrained_model.to(device)
                    hybrid_model = HybridModel(pretrained_model, vib_classifier, device, fc_name=fc_name)
                    hybrid_model.freeze_base()
                    hybrid_model.to(device)
                    untargeted_accuracies, untargeted_examples, untargeted_total_succesful_attacks_list, targeted_accuracies, targeted_examples, targeted_total_succesful_attacks_list, avg_l2_dist_for_sx_targeted_attack = attack_and_eval(
                        hybrid_model, device, test_data_loader, target_label, epsilons, mean=transformation_mean, std=transformation_std)
                    results_dict[run_name] = {
                        'dict_name': pkl_name,
                        'vib_classifier': vib_classifier.to('cpu'),
                        'beta': beta,
                        'test_accuracy': test_accuracy,
                        'fgs_epsilons': epsilons,
                        'untargeted_accuracies': untargeted_accuracies,
                        'untargeted_total_succesful_attacks_list': untargeted_total_succesful_attacks_list,
                        'untargeted_examples': untargeted_examples,
                        'targeted_accuracies': targeted_accuracies,
                        'targeted_total_succesful_attacks_list': targeted_total_succesful_attacks_list,
                        'targeted_examples': targeted_examples,
                        'avg_l2_dist_for_sx_targeted_attack': avg_l2_dist_for_sx_targeted_attack,
                    }
                    print(f'\n\
                        ###### Run summary: beta={beta} ######\n\
                        test acc: {test_accuracy}\n\
                        untargeted succesful attacks at eps={epsilons[0]}: {untargeted_total_succesful_attacks_list[0]}\n\
                        untargeted succesful attacks at eps={epsilons[-1]}: {untargeted_total_succesful_attacks_list[-1]}\n\
                        untargeted acc at eps={epsilons[0]}: {untargeted_accuracies[0]}\n\
                        untargeted acc at eps={epsilons[-1]}: {untargeted_accuracies[-1]}\n\
                        targeted succesful attacks at eps={epsilons[0]}: {targeted_total_succesful_attacks_list[0]}\n\
                        targeted succesful attacks at eps={epsilons[-1]}: {targeted_total_succesful_attacks_list[-1]}\n\
                        targeted acc at eps={epsilons[0]}: {targeted_accuracies[0]}\n\
                        targeted acc at eps={epsilons[-1]}: {targeted_accuracies[-1]}\n\
                        avg l2 distance for succesful cw targeted attack: {avg_l2_dist_for_sx_targeted_attack}\n\
                        ')

            except ValueError as e:
                raise e
                print(f"Exception occured: exploding gradient: {e}\n skipping...")
                continue
            finally:
                wandb_run.finish()

            with open(save_path, 'wb') as f:
                pickle.dump(results_dict, f)
            print(f'Saved dict to {save_path}')



# TEXTUAL_DATASETS = ('ag_news', 'imdb', 'cola', 'mnli')  # 'mr', 'rte'

# train_and_eval_cdlvm('imdb', kl_rate_loss=False, clip_grad=False, clip_loss=True, loss_type='vub', num_minibatches=1, betas=[0.1], num_epochs=1)
# train_and_eval_cdlvm('cola', kl_rate_loss=False, clip_grad=False, clip_loss=True, loss_type='vub')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-class", type=str, default="mnist", help="Kind of dataset to use: mnist, cifar, imagenet or yelp")
    parser.add_argument("--betas", nargs='+', type=float, default=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], help="Betas to use for VIB or VUB")
    parser.add_argument("--epsilons", nargs='+', type=float, default=[0.1, 0.35, 0.4, 0.45, 0.5], help="Epsilons to use for FGSM")
    parser.add_argument("--loss-type", type=str, default="vib", help="Which loss function to use: Either VIB, VUB or PPO or Vanilla")
    parser.add_argument("--kl-rate-loss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Use KLD instead of entropy in first part of rate term in VUB")
    parser.add_argument("--clip-grad", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Clip gradient")
    parser.add_argument("--clip-loss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Clip rate term in loss function")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--num-minibatches", type=int, default=1, help="Number of minibatches")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs per beta")
    parser.add_argument("--num-epochs", type=int, default=0, help="Overide number of epochs")
    parser.add_argument("--text-attack-type", type=str, default="BAEGarg2019", help="Type of attack to use on textual models")
    args = parser.parse_args()
    if args.loss_type == 'ppo' and (args.num_minibatches < 2):
        raise ValueError
    return args

if __name__ == "__main__":
    args = parse_args()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    train_and_eval_cdlvm(data_class=args.data_class, betas=args.betas, epsilons=args.epsilons,
                         loss_type=args.loss_type, kl_rate_loss=args.kl_rate_loss, clip_grad=args.clip_grad,
                         clip_loss=args.clip_loss, num_minibatches=args.num_minibatches, num_runs=args.num_runs,
                         num_epochs=args.num_epochs, text_attack_type=args.text_attack_type)
