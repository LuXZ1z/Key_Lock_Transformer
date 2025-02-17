import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
import seaborn as sns
import random

# Dictionary mapping DNA sequences to keys
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

class CustomSchedule(nn.Module):
    """Custom learning rate scheduler for transformer models."""
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.step = 1.0

    def __call__(self):
        """Compute the current learning rate."""
        arg1 = self.step ** -0.5
        arg2 = self.step * (self.warmup_steps ** -1.5)
        self.step += 1.0
        return (self.d_model ** -0.5) * min(arg1, arg2)


def get_timestamp(format: str = "%y%m%d-%H%M%S"):
    """Get the current timestamp in the specified format."""
    return datetime.now(timezone(timedelta(hours=8))).strftime(format)


def drawPicSide(x_data, y_data, x_label, y_label, file_name):
    """
    Draw a scatter plot with marginal distributions.

    Args:
        x_data (array-like): Data for the x-axis.
        y_data (array-like): Data for the y-axis.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        file_name (str): Name of the file to save the plot.
    """
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_axes([0.1, 0.1, 0.65, 0.65])  # Main scatter plot
    ax2 = fig.add_axes([0.1, 0.85, 0.7, 0.1])   # Top marginal distribution
    ax3 = fig.add_axes([0.85, 0.1, 0.1, 0.7])   # Right marginal distribution

    # Plot scatter and diagonal line
    range_max = np.maximum(np.max(x_data), np.max(y_data))
    range_min = np.minimum(np.min(x_data), np.min(y_data))
    xx = [np.round(_, 1) for _ in np.arange(range_min - 1, range_max + 1, 1)]
    ax1.set_xlim(range_min - 1, range_max + 1)
    ax1.set_ylim(range_min - 1, range_max + 1)
    ax1.set_xlabel(x_label, fontproperties='Times New Roman', fontsize=15)
    ax1.set_ylabel(y_label, fontproperties='Times New Roman', fontsize=15)
    scatter = ax1.scatter(x_data, y_data, s=0.5, label='Data Points')
    ax1.plot(xx, xx, color='silver')

    # Set tick fonts and sizes
    for ax in [ax1, ax2, ax3]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        ax.tick_params(labelsize=15)

    ax1.legend(loc='lower right', fontsize=12)

    # Plot marginal distributions
    sns.kdeplot(x=x_data, ax=ax2)
    ax2.set_ylabel('Density', fontproperties='Times New Roman', fontsize=15)

    sns.kdeplot(y=y_data, ax=ax3)
    ax3.set_xlabel('Density', fontproperties='Times New Roman', fontsize=15)

    plt.savefig(file_name)
    plt.clf()


def split_data(file_path, file_dir):
    """
    Split dataset into training and testing sets.

    Args:
        file_path (str): Path to the input CSV file.
        file_dir (str): Directory to save the split datasets.
    """
    data = pd.read_csv(file_path)
    random_number = random.randint(0, 4294967295)
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=random_number)

    print("Training set size:", len(train_set))
    print("Testing set size:", len(test_set))

    train_path = os.path.join(file_dir, 'train_set.csv')
    test_path = os.path.join(file_dir, 'test_set.csv')
    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)


def draw_train_test_info(list1, list2, name, file_path, epochs):
    """
    Plot training and testing metrics (e.g., loss or Pearson correlation).

    Args:
        list1 (list): Training metric values.
        list2 (list): Validation metric values.
        name (str): Name of the metric ('loss' or 'pearson').
        file_path (str): Directory to save the plot.
        epochs (int): Number of epochs.
    """
    plt.figure(figsize=(10, 6))
    x = range(1, epochs + 1)
    y1 = list1
    y2 = list2

    if name == "loss":
        plt.title('Loss', fontsize=20)
        line1, = plt.plot(x, y1)
        line2, = plt.plot(x, y2)
        plt.legend(handles=[line1, line2], labels=["train", "val"], loc="upper right", fontsize=15)
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.grid()
        plt.savefig(file_path + "/loss.png")
    elif name == "pearson":
        plt.title('Pearson', fontsize=20)
        line1, = plt.plot(x, y1)
        line2, = plt.plot(x, y2)
        plt.legend(handles=[line1, line2], labels=["train", "val"], loc="lower right", fontsize=15)
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('pearson', fontsize=20)
        plt.grid()
        plt.savefig(file_path + "/pearson.png")
    plt.clf()


def save_to_file(file_path, file_name, *args):
    """
    Save multiple lists to a CSV file.

    Args:
        file_path (str): Directory to save the file.
        file_name (str): Name of the file.
        *args (lists): Lists to be saved.
    """
    data = [list(lst) for lst in args]
    df = pd.DataFrame(data)
    full_path = f"{file_path}/{file_name}"
    df.to_csv(full_path, index=False)


def complement(base):
    """Return the complement of a DNA base."""
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return complement_dict[base]


def reverse_complement(sequence):
    """Return the reverse complement of a DNA sequence."""
    return ''.join(complement(base) for base in reversed(sequence))


def levenshtein_distance(seq1, seq2):
    """
    Calculate the Levenshtein distance between two sequences.

    Args:
        seq1 (str): First sequence.
        seq2 (str): Second sequence.

    Returns:
        float: Levenshtein distance.
    """
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def filter_invalid_values(array1, array2):
    """
    Filter out NaN and Inf values from two arrays.

    Args:
        array1 (array-like): First array.
        array2 (array-like): Second array.

    Returns:
        tuple: Filtered arrays.
    """
    array1 = np.array(array1)
    array2 = np.array(array2)
    mask = ~np.isnan(array1) & ~np.isnan(array2) & ~np.isinf(array1) & ~np.isinf(array2)
    return array1[mask], array2[mask]