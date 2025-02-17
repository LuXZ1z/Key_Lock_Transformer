import pickle
import random
import re
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from config import Config
from dataloader import LoadDNA_Infer_Dataset
import seaborn as sns
from model import KLModel_Plus
import os
import time
from copy import deepcopy

import matplotlib

from utils import drawPicSide, reverse_complement, levenshtein_distance, filter_invalid_values

matplotlib.use('Agg')  # Use Agg backend for rendering

# Mapping of keys to their IDs
keys = {
    "GGCATTAT": "Key_1",
    "ACGTTTGA": "Key_3",
    "TGAAGGAA": "Key_4",
    "GAACAATC": "Key_5",
    "CCATTATA": "Key_6",
    "TGTCGGTC": "Key_7",
    "ACGCTGTA": "Key_8",
    "TGCATCTC": "Key_9",
    "GCCTATGT": "Key_10",
    "GCTTGTTA": "Key_11",
    "GGAGGACT": "Key_12",
    "CGAAGGTA": "Key_13",
    "CAACACTT": "Key_14",
    "GGTCGACT": "Key_15",
    "CCCCCTTT": "Key_16",
    "AAAAGTGT": "Key_17",
    "CATCGGAA": "Key_18",
    "TGTATATC": "Key_19",
    "ACGATTTC": "Key_20",
    "ATAAATTT": "Key_21",
    "ACGAACTC": "Key_22",
    "CGTAAGTC": "Key_23",
    "GACCTACT": "Key_24",
    "AGCAGGTC": "Key_25",
    "ATGCAATA": "Key_26",
    "TTTACTAT": "Key_27",
    "AAGGACTA": "Key_28",
    "CAAAGGTA": "Key_29",
    "GCTGCAAA": "Key_30",
    "GCACCATT": "Key_31",
    "TAACTTAC": "Key_32",
    "GAGGAATC": "Key_33",
}

def run_infer(config, file_path, output_name, model_name):
    """
    Runs inference using the provided configuration and model.

    Args:
        config (Config): Configuration object.
        file_path (str): Path to the input dataset.
        output_name (str): Name for the output file.
        model_name (str): Name of the model directory.
    """
    data_loader = LoadDNA_Infer_Dataset(
        batch_size=config.batch_size,
        k_mer1=config.k_mer1,
        k_mer2=config.k_mer2,
        label=True,
        pretrained_emb=True
    )

    data_iter = data_loader.load_data(file_path)

    model = KLModel_Plus(
        d_model=config.d_model,
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
        max_sen_len=2 * (config.max_sen_len1 + 1 - config.k_mer1)
    )

    output_dir = os.path.join(model_name, 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model_path = os.path.join(model_name, 'model.pt')
    loaded_paras = torch.load(model_path)
    model.load_state_dict(loaded_paras)

    print("## Successfully loaded the model, performing inference......")
    model = model.to(config.device)
    infer(data_iter, model, config.device, file_path, output_dir, output_name)


def infer(data_iter, model, device, file_path, output_dir, output_name):
    """
    Performs inference on the dataset with labels.

    Args:
        data_iter (DataLoader): Data iterator.
        model (nn.Module): Model to use for inference.
        device (torch.device): Device to run the model on.
        file_path (str): Path to the input dataset.
        output_dir (str): Directory to save the output.
        output_name (str): Name for the output file.
    """
    model.eval()
    with torch.no_grad():
        predict_label_list = []
        true_label_list = []
        for idx, (x1, x2, label) in enumerate(tqdm(data_iter)):
            x1 = x1.to(device)
            x2 = x2.to(device)
            label = label.to(device)

            logits = model(x1, x2)
            logits = logits.squeeze(dim=1)
            predict_label_list.extend(logits.cpu().detach().numpy())
            true_label_list.extend(label.cpu().detach().numpy())

        true_label_list, predict_label_list = filter_invalid_values(true_label_list, predict_label_list)
        acc = pearsonr(true_label_list, predict_label_list)[0]
        acc_spearman = spearmanr(true_label_list, predict_label_list)[0]

    drawPicSide(true_label_list, predict_label_list, 'true_depth', 'predict_depth',
                f'{output_dir}/output.png')

    df = pd.read_csv(file_path)

    def split_sequence(sequence):
        part1 = sequence[:8]
        part2 = sequence[8:16]
        return part1, part2

    df[['lock', 'key']] = df['seq'].apply(split_sequence).apply(pd.Series)

    df['Key_ID'] = df['key'].map(keys)
    df['key_rc'] = df['key'].apply(reverse_complement)
    df['edit_distance'] = df.apply(lambda row: levenshtein_distance(row['lock'], row['key_rc']), axis=1)

    lock_list = df['lock'].tolist()
    key_list = df['key'].tolist()
    key_info_list = df['Key_ID'].tolist()
    key_rc_list = df['key_rc'].tolist()
    ed_dist_list = df['edit_distance'].tolist()

    ids = range(len(df['seq']))
    data = {
        'seq': df['seq'],
        'key': key_list,
        'lock': lock_list,
        'logits': predict_label_list,
        'label': true_label_list,
        'Key_ID': key_info_list,
        'key_rc': key_rc_list,
        'ed_dist': ed_dist_list,
        'id': ids
    }
    df = pd.DataFrame(data)

    df.to_csv(f'{output_dir}/{output_name}.csv', index=False)
    print(f"Inference results saved to {output_dir}/")
    print(f"Pearson correlation: {acc}")
    print(f"Spearman correlation: {acc_spearman}")


def infer_no_label(data_iter, model, device, file_path, output_dir, output_name):
    """
    Performs inference on the dataset without labels.

    Args:
        data_iter (DataLoader): Data iterator.
        model (nn.Module): Model to use for inference.
        device (torch.device): Device to run the model on.
        file_path (str): Path to the input dataset.
        output_dir (str): Directory to save the output.
        output_name (str): Name for the output file.
    """
    model.eval()
    with torch.no_grad():
        predict_label_list = []
        for idx, (x1, x2) in enumerate(tqdm(data_iter)):
            x1 = x1.to(device)
            x2 = x2.to(device)

            logits = model(x1, x2)
            logits = logits.squeeze(dim=1)
            predict_label_list.extend(logits.cpu().detach().numpy())

    df = pd.read_csv(file_path)

    def split_sequence(sequence):
        part1 = sequence[:8]
        part2 = sequence[8:16]
        return part1, part2

    df[['lock', 'key']] = df['seq'].apply(split_sequence).apply(pd.Series)

    df['Key_ID'] = df['key'].map(keys)
    df['key_rc'] = df['key'].apply(reverse_complement)
    df['edit_distance'] = df.apply(lambda row: levenshtein_distance(row['lock'], row['key_rc']), axis=1)

    lock_list = df['lock'].tolist()
    key_list = df['key'].tolist()
    key_info_list = df['Key_ID'].tolist()
    key_rc_list = df['key_rc'].tolist()
    ed_dist_list = df['edit_distance'].tolist()

    ids = range(len(df['seq']))
    data = {
        'seq': df['seq'],
        'key': key_list,
        'lock': lock_list,
        'logits': predict_label_list,
        'Key_ID': key_info_list,
        'key_rc': key_rc_list,
        'ed_dist': ed_dist_list,
        'id': ids
    }
    df = pd.DataFrame(data)

    df.to_csv(f'{output_dir}/{output_name}.csv', index=False)
    print(f"Inference results saved to {output_dir}/")


if __name__ == '__main__':
    config = Config()
    file_path = r"./input/original_data/seq_depth_mini.csv"
    model_name = 'xxx'
    model_dir = os.path.join(config.model_save_dir, model_name)
    run_infer(config, file_path, f"xxx_{model_name}", model_dir)