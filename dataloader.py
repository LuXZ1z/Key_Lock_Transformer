from collections import Counter
import numpy as np
import pandas as pd
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
import torch
import re
from tqdm import tqdm

from embedding import dna_sequence_to_k_mer

def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    Pads sequences in a list to the same length.

    Args:
        sequences (list of Tensor): List of sequences to pad.
        batch_first (bool): Whether to put batch size first.
        max_len (int): Maximum length to pad to.
        padding_value (int): Value to use for padding.

    Returns:
        Tensor: Padded sequences.
    """
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    length = max_len
    max_len = max([s.size(0) for s in sequences])
    if length is not None:
        max_len = max(length, max_len)
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor

class LoadDNADataset:
    """
    Loads and preprocesses DNA sequence data.
    """

    def __init__(self, batch_size=20, max_sen_len='same', k_mer1=1, k_mer2=1, pretrained_emb=False):
        """
        Initializes the dataset loader.

        Args:
            batch_size (int): Batch size for DataLoader.
            max_sen_len (str or int): Maximum sentence length. 'same' means use the longest sequence in the dataset.
            k_mer1 (int): K-mer size for the first sequence.
            k_mer2 (int): K-mer size for the second sequence.
            pretrained_emb (bool): Whether to use pre-trained embeddings.
        """
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        self.k_mer1 = k_mer1
        self.k_mer2 = k_mer2
        self.pretrained_emb = pretrained_emb

    def data_process(self, file_path):
        """
        Processes the DNA sequence dataset.

        Args:
            file_path (str): Path to the dataset.

        Returns:
            list: List of processed data.
        """
        df = pd.read_csv(file_path)
        sequence_column = 'seq'
        activity_column = 'sum'

        def split_sequence(sequence):
            part1 = sequence[:8]
            part2 = sequence[8:16]
            return part1, part2

        df[['lock', 'key']] = df[sequence_column].apply(split_sequence).apply(pd.Series)
        df[activity_column] = df[activity_column].astype(float)

        lock_list = df['lock'].tolist()
        key_list = df['key'].tolist()

        x1 = dna_sequence_to_k_mer(lock_list, self.k_mer1, self.pretrained_emb)
        x2 = dna_sequence_to_k_mer(key_list, self.k_mer2, self.pretrained_emb)

        activity_mean_list = df[activity_column].tolist()
        activity_mean_log = np.log2([x_i + 1 for x_i in activity_mean_list])

        combined_data = [(torch.tensor(x1_i, dtype=torch.long), torch.tensor(x2_i, dtype=torch.long),
                          torch.tensor(activity_mean_i, dtype=torch.float)) for x1_i, x2_i, activity_mean_i in
                         zip(x1, x2, activity_mean_log)]

        return combined_data

    def load_data(self, data_file_paths):
        """
        Loads the DNA sequence dataset.

        Args:
            data_file_paths (str): Path to the dataset.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        data = self.data_process(data_file_paths)
        data_iter = DataLoader(data, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)
        return data_iter

    def generate_batch(self, data_batch):
        """
        Generates a batch of data.

        Args:
            data_batch (list): List of data samples.

        Returns:
            Tensor: Batch of first sequences.
            Tensor: Batch of second sequences.
            Tensor: Batch of labels.
        """
        batch_x1, batch_x2, batch_label = [], [], []
        for (x1, x2, label) in data_batch:
            batch_x1.append(x1)
            batch_x2.append(x2)
            batch_label.append(label)
        batch_x1 = pad_sequence(batch_x1, batch_first=False)
        batch_x2 = pad_sequence(batch_x2, batch_first=False)
        batch_label = torch.tensor(batch_label, dtype=torch.float)
        return batch_x1, batch_x2, batch_label


class LoadDNA_Infer_Dataset:
    """
    Loads and preprocesses DNA sequence data for inference.
    """

    def __init__(self, data_path=None, batch_size=20, max_sen_len='same', k_mer1=1, k_mer2=1, pretrained_emb=False,
                 label=False):
        """
        Initializes the dataset loader.

        Args:
            data_path (str): Path to the dataset.
            batch_size (int): Batch size for DataLoader.
            max_sen_len (str or int): Maximum sentence length. 'same' means use the longest sequence in the dataset.
            k_mer1 (int): K-mer size for the first sequence.
            k_mer2 (int): K-mer size for the second sequence.
            pretrained_emb (bool): Whether to use pre-trained embeddings.
            label (bool): Whether to include labels in the dataset.
        """
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        self.k_mer1 = k_mer1
        self.k_mer2 = k_mer2
        self.label_en = label
        self.pretrained_emb = pretrained_emb

    def data_process(self, file_path, label=False):
        """
        Processes the DNA sequence dataset.

        Args:
            file_path (str): Path to the dataset.
            label (bool): Whether to include labels in the dataset.

        Returns:
            list: List of processed data.
        """
        df = pd.read_csv(file_path)
        sequence_column = 'seq'

        def split_sequence(sequence):
            part1 = sequence[:8]
            part2 = sequence[8:16]
            return part1, part2

        df[['lock', 'key']] = df[sequence_column].apply(split_sequence).apply(pd.Series)

        lock_list = df['lock'].tolist()
        key_list = df['key'].tolist()

        x1 = dna_sequence_to_k_mer(lock_list, self.k_mer1, self.pretrained_emb)
        x2 = dna_sequence_to_k_mer(key_list, self.k_mer2, self.pretrained_emb)

        if not label:
            combined_seqx = [(torch.tensor(x1_i, dtype=torch.long), torch.tensor(x2_i, dtype=torch.long)) for x1_i, x2_i
                             in zip(x1, x2)]
            return combined_seqx
        else:
            activity_column = 'sum'
            df[activity_column] = df[activity_column].astype(float)
            activity_mean_list = df[activity_column].tolist()
            activity_mean_log = np.log2([x_i + 1 for x_i in activity_mean_list])
            combined_data = [(torch.tensor(x1_i, dtype=torch.long), torch.tensor(x2_i, dtype=torch.long),
                              torch.tensor(activity_mean_i, dtype=torch.float)) for x1_i, x2_i, activity_mean_i in
                             zip(x1, x2, activity_mean_log)]
            return combined_data

    def load_data(self, data_path):
        """
        Loads the DNA sequence dataset.

        Args:
            data_path (str): Path to the dataset.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        data = self.data_process(data_path, label=self.label_en)
        if self.label_en:
            data_iter = DataLoader(data, batch_size=self.batch_size, shuffle=False,
                                   collate_fn=self.generate_batch_label)
        else:
            data_iter = DataLoader(data, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)
        return data_iter

    def generate_batch(self, data_batch):
        """
        Generates a batch of data.

        Args:
            data_batch (list): List of data samples.

        Returns:
            Tensor: Batch of first sequences.
            Tensor: Batch of second sequences.
        """
        batch_x1, batch_x2 = [], []
        for (x1, x2) in data_batch:
            batch_x1.append(x1)
            batch_x2.append(x2)
        batch_x1 = pad_sequence(batch_x1, batch_first=False)
        batch_x2 = pad_sequence(batch_x2, batch_first=False)
        return batch_x1, batch_x2

    def generate_batch_label(self, data_batch):
        """
        Generates a batch of data with labels.

        Args:
            data_batch (list): List of data samples.

        Returns:
            Tensor: Batch of first sequences.
            Tensor: Batch of second sequences.
            Tensor: Batch of labels.
        """
        batch_x1, batch_x2, batch_label = [], [], []
        for (x1, x2, label) in data_batch:
            batch_x1.append(x1)
            batch_x2.append(x2)
            batch_label.append(label)
        batch_x1 = pad_sequence(batch_x1, batch_first=False)
        batch_x2 = pad_sequence(batch_x2, batch_first=False)
        batch_label = torch.tensor(batch_label, dtype=torch.float)
        return batch_x1, batch_x2, batch_label


if __name__ == '__main__':
    file_path = './input/original_data/seq_depth_mini_2.csv'
    data_loader = LoadDNADataset(batch_size=2, k_mer1=3, k_mer2=3)
    data_loader.data_process(file_path)

    data_loader2 = LoadDNA_Infer_Dataset(batch_size=2, k_mer1=3, k_mer2=3, label=True)
    data_iter = data_loader.load_data(file_path)
    data_iter2 = data_loader2.load_data(file_path)
    for idx, (x1, x2, label) in enumerate(data_iter):
        print(len(x1))