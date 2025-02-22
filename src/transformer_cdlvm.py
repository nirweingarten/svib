import os
import torch
from transformers import AutoTokenizer
import torch.nn as nn
from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper
from transformers import PreTrainedModel
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from ceb_utils import BackwardsEncoder
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import DeepWordBugGao2018, PWWSRen2019
from textattack import Attacker, AttackArgs
from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
)


ATTACK_TYPES = {
    'DeepWordBugGao2018': DeepWordBugGao2018
}

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


class TransformerVIB(nn.Module):
    """
    Transformer classifier with stochastic layer and KL regularization
    This can be optimized with either VIB or VUB
    """

    def __init__(self, hidden_size, output_size, device, is_ceb=False, trick=''):
        super(TransformerVIB, self).__init__()
        self.device = device
        self.description = 'Vanilla IB VAE as per the VIB paper'
        self.hidden_size = hidden_size
        self.trick = trick
        if 'noise' in trick:
            self.k = hidden_size
        else:
            self.k = hidden_size // 2
        self.output_size = output_size
        self.train_loss = []
        self.test_loss = []

        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.ReLU() removed as per ceb
        )
        self.classifier = nn.Linear(self.k, output_size)
        # These are cheats to make 'drill' save everythung we need in one pickle
        self.softplus = F.softplus
        self.normal = torch.normal
        self.Normal = Normal
        self.is_ceb = is_ceb
        if is_ceb:
            self.backwards_encoder = BackwardsEncoder(self.k, output_size, device)

        # Xavier initialization
        for _, module in self._modules.items():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(
                    module.weight, gain=nn.init.calculate_gain('relu'))
                module.bias.data.zero_()
                continue
            if isinstance(module, BackwardsEncoder):
                nn.init.xavier_uniform_(module.encoder.weight, gain=nn.init.calculate_gain('relu'))
                module.encoder.bias.data.zero_()
                continue
            for layer in module:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(
                        layer.weight, gain=nn.init.calculate_gain('relu'))
                    layer.bias.data.zero_()

    def reparametrize(self, mu, std):
        """
        Performs reparameterization trick z = mu + epsilon * std
        Where epsilon~N(0,1)
        """
        mu = mu.to(self.device).expand(1, * mu.size())
        std = std.to(self.device).expand(1, *std.size())
        eps = self.normal(0, 1, size=std.size()).to(self.device)
        return mu + eps * std

    def forward(self, x, y=None):
        x = x.to(self.device)
        z_params = self.encoder(x)
        if 'noise' in self.trick:
            mu = z_params
            std = torch.ones_like(mu)
        else:
            mu = z_params[:, :self.k]
            std = self.softplus(z_params[:, self.k:] - 1, beta=1)
        if self.training:
            z = self.reparametrize(mu, std)
        else:
            z = mu.clone().unsqueeze(0)
        # n = self.Normal(mu, std)
        e_z_x_dist = Independent(Normal(mu, std), 1)
        h_z_x = -e_z_x_dist.log_prob(z.squeeze(0))  # These may be positive as this is a PDF

        logits = self.classifier(z)
        if self.is_ceb and self.training:
            b_z_y_dist, backwards_mu = self.backwards_encoder(y)
            h_z_y = -b_z_y_dist.log_prob(z.squeeze(0))
            return (mu, std), h_z_x, logits, h_z_y, b_z_y_dist, e_z_x_dist
        else:
            return (mu, std), h_z_x, logits


class TransformerHybridModel(nn.Module):
    """
    Head is a pretrained model, classifier is stochastic transformer
    fc_name should be 'classifier' for Bert (imdb)
    """

    def __init__(self, base_model, vib_model, device, fc_name, return_only_logits=False):
        super(TransformerHybridModel, self).__init__()
        self.device = device
        self.base_model = base_model
        setattr(self.base_model, fc_name, torch.nn.Identity())
        self.base_model.classifier = torch.nn.Identity()
        self.vib_model = vib_model
        self.base_model = self.base_model.to(device)
        self.vib_model = self.vib_model.to(device)
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

    def forward(self, **kwargs):
        # This is not really logits, only called that way cause we've changed the final layer to identity
        encoded = self.base_model(kwargs['input_ids'].to(self.device)).logits
        (mu, std), log_probs, logits = self.vib_model(encoded)
        if self.return_only_logits:
            return logits.squeeze(0)
        else:
            return ((mu, std), log_probs, logits)


class TransformerAdaptor(transformers.PreTrainedModel):
    """
    Adapts between a TransformerHybridModel to a HuggingFaceModelWrapper s.t. we can use the textattack package
    """

    def __init__(self, hybrid_model):
        super(TransformerAdaptor, self).__init__(
            hybrid_model.base_model.config)
        self.hybrid_model = hybrid_model
        # Cheat to overload drill pickle
        self.SequenceClassifierOutput = SequenceClassifierOutput

    def forward(self, **kwargs):
        if self.hybrid_model.return_only_logits:
            logits = self.hybrid_model(**kwargs)
        else:
            ((_, _), _, logits) = self.hybrid_model(**kwargs)
        return self.SequenceClassifierOutput(logits=logits[0])


def encode(examples):
    if 'text' in examples.keys():
        key = 'text'
    elif 'sentence' in examples.keys():
        key = 'sentence'
    return tokenizer(examples[key], truncation=True, padding='max_length', max_length=512)


def attack_model(attack, dataset):
    attack_args = AttackArgs(
        num_examples=200,
        disable_stdout=True,
        silent=True,
        enable_advance_metrics=False
    )
    attacker = Attacker(attack, dataset, attack_args)
    results_iterable = attacker.attack_dataset()
    attack_log_manager = attacker.attack_log_manager
    attack_log_manager.log_summary()
    attack_success_stats = AttackSuccessRate().calculate(attack_log_manager.results)
    words_perturbed_stats = WordsPerturbed().calculate(attack_log_manager.results)
    attack_query_stats = AttackQueries().calculate(attack_log_manager.results)
    acc_under_attack = str(attack_success_stats["attack_accuracy_perc"])
    avg_pertrubed_words_prct = str(words_perturbed_stats["avg_word_perturbed_perc"])
    attack_success_rate = attack_success_stats['attack_success_rate']
    original_acc = attack_success_stats["original_accuracy"]
    return original_acc, [acc_under_attack], avg_pertrubed_words_prct, attack_success_rate


def text_attacks(hybrid_model, dataset_name, device):
    """
    Performs deep word bug and pwws untargeted blackbox attacks using the textattack API
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # supress noisy tf logs
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model_adaptor = TransformerAdaptor(hybrid_model)
    model_adaptor = model_adaptor.to(device)
    model_wrapper = HuggingFaceModelWrapper(model_adaptor, tokenizer)
    model_wrapper.to(device)

    if dataset_name in ('ag_news', 'imdb'):
        dataset = HuggingFaceDataset(dataset_name, None, "test")
    else:
        raise NotImplementedError

    deep_word_bug_attack = DeepWordBugGao2018.build(model_wrapper)
    pwws_attack = PWWSRen2019.build(model_wrapper)
    
    print('### Running Deep Word Bug attack')
    original_acc, deep_word_acc_under_attack, deep_word_avg_pertrubed_words_prct, deep_word_attack_success_rate = attack_model(deep_word_bug_attack, dataset)

    print('### Running PWWS attack')
    _, pwws_acc_under_attack, pwws_avg_pertrubed_words_prct, pwws_attack_success_rate = attack_model(pwws_attack, dataset)

    return original_acc, deep_word_acc_under_attack, deep_word_avg_pertrubed_words_prct, deep_word_attack_success_rate, pwws_acc_under_attack, pwws_avg_pertrubed_words_prct, pwws_attack_success_rate
