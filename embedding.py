import csv
import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the sequence tokens.
    Uses sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model, dropout=0.1, max_len=25):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # Positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Reshape for batch processing
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        Args:
            x (Tensor): Input tensor of shape [seq_len, batch_size, d_model]
        Returns:
            Tensor: Output tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding using an embedding layer.
    """
    def __init__(self, d_model, dropout=0.1, max_len=25):
        super(LearnablePositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Adds learnable positional encoding to the input tensor.
        Args:
            x (Tensor): Input tensor of shape [seq_len, batch_size, d_model]
        Returns:
            Tensor: Output tensor with positional encoding added.
        """
        positions = torch.arange(0, x.size(0), dtype=torch.long, device=x.device).unsqueeze(1)
        return self.dropout(x + self.embedding(positions))


class TokenEmbedding(nn.Module):
    """
    Embeds tokens into a dense vector space.
    """
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        """
        Embeds the input tokens.
        Args:
            tokens (Tensor): Input tokens of shape [seq_len, batch_size]
        Returns:
            Tensor: Embedded tokens of shape [seq_len, batch_size, emb_size]
        """
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def dna_sequence_to_numbers(dna_sequences):
    """
    Converts DNA sequences to numerical representations.
    Args:
        dna_sequences (list): List of DNA sequences.
    Returns:
        list: List of numerical sequences.
    """
    base_mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    number_sequences = []
    for dna_sequence in dna_sequences:
        number_sequence = [base_mapping[base] for base in dna_sequence]
        number_sequences.append(number_sequence)
    return number_sequences


def get_index_by_word(word_list, vocab_index):
    """
    Retrieves indices for words from a vocabulary index.
    Args:
        word_list (list): List of words.
        vocab_index (dict): Vocabulary index mapping words to indices.
    Returns:
        list: List of indices.
    """
    indices = []
    for word in word_list:
        index = vocab_index.get(word)
        if index is not None:
            indices.append(index)
        else:
            print(f"Word '{word}' not found in vocabulary.")
    return indices


def load_word_indices_from_csv(file_path):
    """
    Loads word indices from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        dict: Dictionary mapping words to indices.
    """
    vocab_index = {}
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header
        for row in reader:
            word = row[0]
            index = int(row[1])
            vocab_index[word] = index
    return vocab_index


def dna_sequence_to_k_mer(dna_sequences, k, pretrained_emb=False):
    """
    Converts DNA sequences to k-mer representations.
    Args:
        dna_sequences (list): List of DNA sequences.
        k (int): Length of k-mers.
        pretrained_emb (bool): Whether to use pre-trained embeddings.
    Returns:
        list: List of k-mer numerical sequences.
    """
    if not pretrained_emb:
        base_mapping = generate_k_mer_mapping(k)
        k_mer_dna = []
        for dna_sequence in dna_sequences:
            k_mers = []
            for i in range(len(dna_sequence) - k + 1):
                k_mer = ''.join(dna_sequence[i:i + k])
                k_mers.append(k_mer)
            k_mers.insert(0, 'CLS')
            k_mers.append('SEP')
            k_mer_dna.append(k_mers)
        number_sequences = []
        for dna_sequence in k_mer_dna:
            number_sequence = [base_mapping[base] for base in dna_sequence]
            number_sequences.append(number_sequence)
        return number_sequences
    else:
        base_mapping = load_word_indices_from_csv('./pretrained/word_indices.csv')
        k_mer_dna = []
        for dna_sequence in dna_sequences:
            k_mers = []
            for i in range(len(dna_sequence) - k + 1):
                k_mer = ''.join(dna_sequence[i:i + k])
                k_mers.append(k_mer)
            k_mer_dna.append(k_mers)
        number_sequences = []
        for dna_sequence in k_mer_dna:
            number_sequence = get_index_by_word(dna_sequence, base_mapping)
            number_sequences.append(number_sequence)
        return number_sequences


def generate_k_mer_mapping(k):
    """
    Generates a mapping of k-mers to indices.
    Args:
        k (int): Length of k-mers.
    Returns:
        dict: Dictionary mapping k-mers to indices.
    """
    bases = ['A', 'T', 'C', 'G']
    base_mapping = {}
    base_mapping['PAD'] = 0
    base_mapping['CLS'] = 4 ** k + 1
    base_mapping['SEP'] = 4 ** k + 2
    count = 0
    for i in range(4 ** k):
        k_mer = ''
        quotient = i
        for j in range(k):
            remainder = quotient % 4
            k_mer += bases[remainder]
            quotient = quotient // 4
        base_mapping[k_mer] = count + 1
        count += 1
    return base_mapping


def load_word2vec_model(filepath):
    """
    Loads a pre-trained Word2Vec model from a file.
    Args:
        filepath (str): Path to the Word2Vec model file.
    Returns:
        tuple: Dictionary of word vectors and vocabulary index.
    """
    word2vec_dict = {}
    word2vec_dict['PAD'] = torch.zeros(100, dtype=torch.float)
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the first line with metadata
            parts = line.split()
            word = parts[0]
            vector = [float(value) for value in parts[1:]]
            word2vec_dict[word] = torch.FloatTensor(vector)
    index_dict = {word: idx for idx, word in enumerate(word2vec_dict.keys())}
    return word2vec_dict, index_dict


if __name__ == '__main__':
    # Example usage
    dna_sequences = [['A', 'T', 'C', 'G', 'A', 'T'], ['A', 'C', 'C', 'A', 'A', 'T']]
    k_mer_sequences = dna_sequence_to_k_mer(dna_sequences, k=3, pretrained_emb=True)
    print(k_mer_sequences)

    token_embedding = TokenEmbedding(vocab_size=66, emb_size=32)
    k_mer_sequences = torch.tensor(k_mer_sequences, dtype=torch.long)
    embedded_sequences = token_embedding(tokens=k_mer_sequences)
    print(embedded_sequences.shape)
    print(embedded_sequences[0])

    pos_embedding = PositionalEncoding(d_model=32)
    embedded_sequences = pos_embedding(x=embedded_sequences)
    print(embedded_sequences[0])
    print(embedded_sequences.shape)