import pickle
import matplotlib
import pandas as pd
import torch.nn as nn
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from BMC_loss import BMCLoss
from config import Config
from dataloader import LoadDNADataset
import os
import time
from copy import deepcopy

from model import KLModel_Plus
from utils import CustomSchedule, get_timestamp, split_data, save_to_file, drawPicSide, draw_train_test_info, \
    filter_invalid_values

# Set matplotlib backend to avoid display issues
matplotlib.use('Agg')

# Get current timestamp for logging and saving
timestamp = get_timestamp()

def train_model(config, timestamp):
    """
    Train the DNA sequence model with the given configuration and timestamp.

    Args:
        config (Config): Configuration object containing model and training settings.
        timestamp (str): Timestamp for logging and saving results.
    """
    # Decide whether to load existing data or create new data loaders
    load_dataset = True
    if not load_dataset:
        try:
            # Load pre-saved training and testing iterators
            with open('./input/saved_train_iter.pkl', 'rb') as f:
                train_iter = pickle.load(f)
            with open('./input/saved_test_iter.pkl', 'rb') as f:
                test_iter = pickle.load(f)
            print("Loaded saved data_loader and iterators.")
        except (FileNotFoundError, EOFError):
            print("Loaded data is not exited")
            exit()
    else:
        # Split dataset and create data loaders
        split_data(config.dataset_file_paths, config.dataset_dir)
        train_data_loader = LoadDNADataset(batch_size=config.batch_size,
                                           k_mer1=config.k_mer1,
                                           k_mer2=config.k_mer2,
                                           pretrained_emb=config.pretrained_emb)
        train_iter = train_data_loader.load_data(
            config.train_corpus_file_paths)

        test_data_loader = LoadDNADataset(batch_size=config.batch_size,
                                          k_mer1=config.k_mer1,
                                          k_mer2=config.k_mer2,
                                          pretrained_emb=config.pretrained_emb)
        test_iter = test_data_loader.load_data(
            config.test_corpus_file_paths)

        # Save data loaders to disk for future use
        with open('./input/saved_train_iter.pkl', 'wb') as f:
            pickle.dump(train_iter, f)
        with open('./input/saved_test_iter.pkl', 'wb') as f:
            pickle.dump(test_iter, f)
        print("Saved data_loader to disk.")

    # Initialize the model with given configuration
    model = KLModel_Plus(d_model=config.d_model,
                         nhead=config.num_head,
                         num_encoder_layers=config.num_encoder_layers,
                         dim_feedforward=config.dim_feedforward,
                         dim_output=config.dim_output,
                         output=config.output,
                         dropout=config.dropout,
                         cnn_channels=config.cnn_channels,
                         kernel_sizes=config.kernel_sizes,
                         k_mer=config.k_mer1,
                         pretrained_emb=config.pretrained_emb,
                         max_sen_len=2 * (config.max_sen_len1 + 1 - config.k_mer1))  # max + 2 - kmer + 1, False adds sep information

    # Initialize model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Create directories for saving model and figures
    save_dir = os.path.join(config.model_save_dir, f'{timestamp}')
    os.mkdir(save_dir)
    save_figs_dir = os.path.join(save_dir, 'figs')
    os.mkdir(save_figs_dir)
    model_save_path = os.path.join(save_dir, 'model.pt')
    model = model.to(config.device)

    # Define loss function and optimizer
    loss_fn = BMCLoss(config.init_noise_sigma)
    learning_rate = CustomSchedule(config.d_model)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.,
                                 betas=(config.beta1, config.beta2),
                                 eps=config.epsilon)
    optimizer.add_param_group({'params': loss_fn.noise_sigma, 'lr': config.sigma_lr, 'name': 'noise_sigma'})

    # Training loop
    model.train()
    max_test_acc = 0
    train_acc_list = []
    train_acc_spearman_list = []
    train_loss_list = []
    test_acc_list = []
    test_acc_spearman_list = []
    test_loss_list = []
    early_stopping_rounds = config.early_stopping_rounds
    no_improvement_count = 0
    train_epoch = 0

    for epoch in range(config.epochs):
        losses = 0
        train_epoch += 1
        start_time = time.time()
        predict_label_list = []
        true_label_list = []

        # Train on each batch
        for idx, (x1, x2, label) in enumerate(tqdm(train_iter)):
            x1 = x1.to(config.device)
            x2 = x2.to(config.device)
            label = label.to(config.device)

            logits = model(x1, x2)
            logits = logits.squeeze(dim=1)

            optimizer.zero_grad()
            loss = loss_fn(logits, label)
            loss.backward()

            # Update learning rate
            lr = learning_rate()
            for p in optimizer.param_groups:
                p['lr'] = lr

            optimizer.step()
            losses += loss.item()

            predict_label_list.extend(logits.cpu().detach().numpy())
            true_label_list.extend(label.cpu().detach().numpy())

        end_time = time.time()
        train_loss = losses / len(train_iter)

        # Filter invalid values and calculate metrics
        true_label_list, predict_label_list = filter_invalid_values(true_label_list, predict_label_list)
        acc = pearsonr(true_label_list, predict_label_list)[0]
        acc_spearman = spearmanr(true_label_list, predict_label_list)[0]

        # Save training results and visualize
        drawPicSide(true_label_list, predict_label_list, 'true_depth', 'predict_depth',
                    f'./cache/{timestamp}/figs/train_{epoch}.png')
        train_acc_list.append(acc)
        train_acc_spearman_list.append(acc_spearman)
        train_loss_list.append(train_loss)

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train pearson: {acc:.3f}, Train spearman: {acc_spearman:.3f}, Epoch time = {(end_time - start_time):.3f}s")

        # Evaluate on test set
        test_acc, test_acc_spearman, test_loss = evaluate(test_iter, model, loss_fn, config.device, epoch)
        print(f"Epoch: {epoch}, Test loss: {test_loss:.3f}, Test pearson: {test_acc:.3f}, Test spearman: {test_acc_spearman:.3f}")

        test_acc_list.append(test_acc)
        test_acc_spearman_list.append(test_acc_spearman)
        test_loss_list.append(test_loss)

        # Check for early stopping
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            no_improvement_count = 0
            state_dict = deepcopy(model.state_dict())
            torch.save(state_dict, model_save_path)
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_rounds:
            print(f"Early stopping after {epoch + 1} epochs due to no improvement.")
            break

    # Save training and testing metrics
    draw_train_test_info(train_acc_list, test_acc_list, 'pearson', save_dir, train_epoch)
    draw_train_test_info(train_loss_list, test_loss_list, 'loss', save_dir, train_epoch)
    save_to_file(save_dir, 'training_result', train_acc_list, train_acc_spearman_list, train_loss_list,
                 test_acc_list, test_acc_spearman_list, test_loss_list)


def evaluate(data_iter, model, loss_fn, device, epoch):
    """
    Evaluate the model on the given dataset.

    Args:
        data_iter: Data iterator for the evaluation dataset.
        model: Model to be evaluated.
        loss_fn: Loss function used for evaluation.
        device: Device to run the evaluation on.
        epoch: Current epoch number.

    Returns:
        acc: Pearson correlation coefficient.
        acc_spearman: Spearman correlation coefficient.
        loss: Average loss on the evaluation dataset.
    """
    model.eval()
    with torch.no_grad():
        predict_label_list = []
        true_label_list = []
        losses = 0

        # Evaluate on each batch
        for idx, (x1, x2, label) in enumerate(data_iter):
            x1 = x1.to(device)
            x2 = x2.to(device)
            label = label.to(device)

            logits = model(x1, x2)
            logits = logits.squeeze(dim=1)
            loss = loss_fn(logits, label)
            losses += loss.item()

            predict_label_list.extend(logits.cpu().detach().numpy())
            true_label_list.extend(label.cpu().detach().numpy())

        # Switch back to training mode
        model.train()

        # Filter invalid values and calculate metrics
        true_label_list, predict_label_list = filter_invalid_values(true_label_list, predict_label_list)
        acc = pearsonr(true_label_list, predict_label_list)[0]
        acc_spearman = spearmanr(true_label_list, predict_label_list)[0]

        # Visualize results
        drawPicSide(true_label_list, predict_label_list, 'true_depth', 'predict_depth',
                    f'./cache/{timestamp}/figs/test_{epoch}.png')

        loss = losses / len(data_iter)
        return acc, acc_spearman, loss


if __name__ == '__main__':
    # Load configuration and start training
    config = Config()
    train_model(config, timestamp)
