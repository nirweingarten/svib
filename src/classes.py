import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from helper import reparametrize


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
        x = x.view(-1, 64 * 7 * 7 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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

    def forward(self, x):
        z_params = self.encoder(x)
        mu = z_params[:, :self.k]
        # softplus transformation (soft relu) and a -1 bias is added
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
