import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from constants import CIFAR_LOGITS_TEST_DATALOADER_PATH, CIFAR_LOGITS_TRAIN_DATALOADER_PATH, DATASET_DIR, DEVICE, EPSILON, IMAGENET_LOGITS_TRAIN_DATALOADER_PATH, IMAGENET_LOGITS_VAL_DATALOADER_PATH, MNIST_LOGITS_TRAIN_DATALOADER_PATH, NP_EPSILON, NUM_WORKERS, TEXTUAL_DATASETS
import cw
from classes import HybridModel, LogitsDataset
import torch.nn.functional as F
from tqdm import tqdm
from constants import MNIST_LOGITS_TEST_DATALOADER_PATH
from transformer_cdlvm import encode
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, MNIST, ImageNet
import os
import pickle


def get_dataloaders(data_class, logits=False):
    if data_class == 'mnist':
        if logits:
            with open(MNIST_LOGITS_TRAIN_DATALOADER_PATH, 'rb') as f:
                train_data_loader = pickle.load(f)
            with open(MNIST_LOGITS_TEST_DATALOADER_PATH, 'rb') as f:
                val_data_loader = pickle.load(f)
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
                train_data_loader = pickle.load(f)
            with open(CIFAR_LOGITS_TEST_DATALOADER_PATH, 'rb') as f:
                val_data_loader = pickle.load(f)
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
                train_data_loader = pickle.load(f)
            with open(IMAGENET_LOGITS_VAL_DATALOADER_PATH, 'rb') as f:
                val_data_loader = pickle.load(f)
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
                train_data_loader = pickle.load(f)
            with open(logits_test_dataloader_path, 'rb') as f:
                val_data_loader = pickle.load(f)
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

    return train_loader, test_loader


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
            if (len(adv_examples) < 5):
                adv_ex = perturbed_data[successful_attack_idx][0].squeeze(
                ).detach().cpu().numpy()
                adv_examples.append((init_pred[successful_attack_idx][0].item(
                ), perturbed_pred[successful_attack_idx][0].item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / relevant_pertrubations
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


def vib_loss(logits, labels, mu, std, beta):
    classification_loss = nn.CrossEntropyLoss()(logits.squeeze(0), labels)  # In torch cross entropy function applies the softmax itself
    normalization_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum()
    return classification_loss + beta * normalization_loss


def reparametrize(mu, std, device):
    """
    Performs reparameterization trick z = mu + epsilon * std
    Where epsilon~N(0,1)
    """
    mu = mu.expand(1, *mu.size())
    std = std.expand(1, *std.size())
    eps = torch.normal(0, 1, size=std.size()).to(device)
    return mu + eps * std


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


def get_multinomial_entropy(logits, epsilon=EPSILON):
    """
    Receives unactivates logits
    epsilon replaces 0 probability that results from torch's low float resolution
    """
    logits = logits.squeeze(0).to(torch.float64)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.clamp(probs, epsilon, 1 - epsilon)
    return (probs * torch.log(1/probs)).sum(dim=-1).mean()


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


def get_logits_dataloader(model, original_loader, batch_size=32, whiten=False, device=DEVICE):
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


def test_model(model, test_data_loader, device=DEVICE):
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
