from datetime import datetime
from distutils.util import strtobool
import argparse
import torch
import numpy as np
from helper import VIB
from constants import EPSILON, TEXTUAL_DATASETS
import numpy as np
import torch
import os
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from helper import LogitsDataset, HybridModel, attack_and_eval, get_dataloaders, get_kld_between_multivariate_gaussians, get_multinomial_entropy, get_multivariate_gaussian_entropy, test_model
from transformer_cdlvm import TransformerVIB, TransformerHybridModel, text_attacks
import wandb
from torch.utils.tensorboard import SummaryWriter


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
                    if clip_loss:
                        max_regularization_value = (abs(beta) * classification_loss).item()
                        min_regularization_value = torch.tensor(0).to(device)
                        minibatch_loss = torch.clamp(rate_term, min=min_regularization_value, max=(max_regularization_value)) + classification_loss - torch.clamp(beta * batch_h_z_y, min=min_regularization_value, max=max_regularization_value)
                    else:
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
        writer.add_scalar(
            "charts/epoch_classification_loss", epoch_classification_loss / len(train_dataloader), e)
        writer.add_scalar(
            "charts/epoch_total_kld", epoch_total_kld / len(train_dataloader), e)
        # Entropy terms cancel out, these deltas are percise
        if e > 0:
            epoch_i_z_x_delta = epoch_h_z_x_array[e] - epoch_h_z_x_array[e - 1]
            epoch_i_z_y_delta = epoch_h_z_y_array[e] - epoch_h_z_y_array[e - 1] 

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
    save_path = f'./result_dicts/{pkl_name}_{formatted_time}.pkl'

    os.environ["WANDB_SILENT"] = "true"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if data_class == 'mnist':
        epochs = 250
        hidden_size = 256
        output_size = 10
        pretrained_path = './pretrained_models/mnist_model.pkl'
        fc_name = 'fc2'
        target_label = 1
        max_grad_norm = 0.001
        transformation_mean = (0, 0, 0)
        transformation_std = (0, 0, 0)
    elif data_class == 'cifar':
        epochs = 100
        hidden_size = 1280
        output_size = 100
        pretrained_path = './pretrained_models/cifar_efficientnet.pkl'
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
        pretrained_path = './pretrained_models/inceptionv3.pkl'
        target_label = 805  # soccer ball
        max_grad_norm = 5
        transformation_mean = (0.485, 0.456, 0.406)
        transformation_std = (0.229, 0.224, 0.225)
    elif data_class in ('yelp', 'imdb', 'cola'):
        fc_name = 'classifier'
        epochs = 200
        hidden_size = 768
        output_size = 2
        pretrained_path = f'./pretrained_models/pretrained_bert_{data_class}.pkl'
        target_label = 0  # Not relevant
        max_grad_norm = 5
    elif data_class.replace('-', '_') == 'mnli':
        fc_name = 'classifier'
        epochs = 200
        hidden_size = 768
        output_size = 3
        pretrained_path = f'./pretrained_models/pretrained_bert_{data_class}.pkl'
        target_label = 0  # Not relevant
        max_grad_norm = 5
    elif data_class.replace('-', '_') == 'ag_news':
        fc_name = 'classifier'
        epochs = 200
        hidden_size = 768
        output_size = 4
        pretrained_path = f'./pretrained_models/pretrained_bert_{data_class}.pkl'
        target_label = 0  # Not relevant
        max_grad_norm = 5
    else:
        raise NotImplementedError
    
    if num_epochs:
        print(f'### Overiding original epochs of {epochs} with user defined {num_epochs}')
        epochs = num_epochs

    logits_train_data_loader, logits_test_data_loader = get_dataloaders(data_class, logits=True)
    _, test_data_loader = get_dataloaders(data_class, logits=False)

    if loss_type == 'vanilla':
        print(f"\n### Evaluating pretrained vanilla model ###")
        pretrained_model = torch.load(pretrained_path)
        pretrained_model.to(device)
        pretrained_model.eval()
        vanilla_run_name = f'vanilla_model_{formatted_time}'
        # wandb_run = wandb.init(
        #     project='dynamic_beta',
        #     entity=None,
        #     sync_tensorboard=True,
        #     config=None,
        #     name=vanilla_run_name,
        #     monitor_gym=False,
        #     save_code=True,)
        writer = SummaryWriter(f"runs/{vanilla_run_name}")
        test_accuracy = test_model(pretrained_model, test_data_loader)
        untargeted_accuracies, untargeted_examples, untargeted_total_succesful_attacks_list, targeted_accuracies, targeted_examples, targeted_total_succesful_attacks_list, avg_l2_dist_for_sx_targeted_attack = attack_and_eval(pretrained_model, device, test_data_loader, target_label, epsilons, mean=transformation_mean, std=transformation_std)
        # wandb_run.finish()
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
            # wandb_run = wandb.init(
            #     project='dynamic_beta',
            #     entity=None,
            #     sync_tensorboard=True,
            #     config=None,
            #     name=run_name,
            #     monitor_gym=False,
            #     save_code=True,)
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
                pass
                # wandb_run.finish()

            with open(save_path, 'wb') as f:
                pickle.dump(results_dict, f)
            print(f'Saved dict to {save_path}')

# For debugging
# train_and_eval_cdlvm('imdb', kl_rate_loss=False, clip_grad=False, clip_loss=True, loss_type='vub', num_minibatches=1, betas=[0.1], num_epochs=1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-class", type=str, default="mnist", help="Kind of dataset to use: mnist, cifar, imagenet or yelp")
    parser.add_argument("--betas", nargs='+', type=float, default=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], help="Betas to use for VIB or VUB")
    parser.add_argument("--epsilons", nargs='+', type=float, default=[0.1, 0.35, 0.4, 0.45, 0.5], help="Epsilons to use for FGSM")
    parser.add_argument("--loss-type", type=str, default="vib", help="Which loss function to use: Either VIB, VUB or PPO or Vanilla")
    parser.add_argument("--kl-rate-loss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Use KLD instead of entropy in first part of rate term in VUB")
    parser.add_argument("--clip-grad", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Clip gradient")
    parser.add_argument("--clip-loss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Clip rate term in loss function")
    parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
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
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    train_and_eval_cdlvm(data_class=args.data_class, betas=args.betas, epsilons=args.epsilons,
                         loss_type=args.loss_type, kl_rate_loss=args.kl_rate_loss, clip_grad=args.clip_grad,
                         clip_loss=args.clip_loss, num_minibatches=args.num_minibatches, num_runs=args.num_runs,
                         num_epochs=args.num_epochs, text_attack_type=args.text_attack_type)
