import torch
import torch.nn as nn
from MyTransformer import MyTransformerEncoder, MyTransformerEncoderLayer
from embedding import PositionalEncoding, TokenEmbedding, load_word2vec_model
import torch.nn.functional as F

class KLModel_Plus(nn.Module):
    """
    A custom model combining Transformer encoder and CNN layers for DNA sequence processing.

    Args:
        vocab_size (int): Vocabulary size for token embedding.
        k_mer (int): K-mer size for DNA sequence processing.
        d_model (int): Dimension of the model.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of layers in the Transformer encoder.
        dim_feedforward (int): Dimension of the feedforward network.
        dim_output (int): Dimension of the output layer.
        output (int): Final output dimension.
        dropout (float): Dropout probability.
        cnn_channels (int): Number of channels in the CNN layers.
        kernel_sizes (list): List of kernel sizes for the CNN layers.
        max_sen_len (int): Maximum sequence length.
        pretrained_emb (bool): Whether to use pre-trained embeddings.
    """
    def __init__(self, vocab_size=None,
                 k_mer=1,
                 d_model=512, nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=2048,
                 dim_output=64,
                 output=1,
                 dropout=0.1,
                 cnn_channels=512,
                 kernel_sizes=[23, 23, 23, 23],
                 max_sen_len=25,
                 pretrained_emb=False):
        super(KLModel_Plus, self).__init__()
        if pretrained_emb:
            print("Loading pre-trained vocabulary:")
            self.vocab, self.vocab_index = load_word2vec_model(
                './pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v')
            vocab_size = len(self.vocab)
            emb_size = next(iter(self.vocab.values())).size(0)
            d_model = emb_size
            embedding_matrix = torch.zeros(vocab_size, emb_size)
            for idx, (word, vector) in enumerate(self.vocab.items()):
                embedding_matrix[idx] = vector
            self.src_token_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            vocab_size = 4 ** k_mer + 3
            self.src_token_embedding = TokenEmbedding(vocab_size=vocab_size, emb_size=d_model)

        self.pos_embedding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_sen_len)
        encoder_layer = MyTransformerEncoderLayer(d_model, nhead,
                                                  dim_feedforward,
                                                  dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = MyTransformerEncoder(encoder_layer,
                                            num_encoder_layers, encoder_norm)
        self.weight_linear = nn.Linear(d_model, d_model, bias=True)
        self.linear_output = nn.Sequential(nn.Linear(d_model, dim_output),
                                           nn.Dropout(0.6),
                                           nn.ReLU(),
                                           nn.Linear(dim_output, output))

        self.conv_list = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=cnn_channels, kernel_size=k),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_sen_len - k + 1)
        ) for k in kernel_sizes])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_channels * len(kernel_sizes), 1)

    def forward(self,
                src1,  # [src_len, batch_size]
                src2,
                src_mask=None,
                src_key_padding_mask=None,  # [batch_size, src_len]
                concat_type='sum'  # How to combine outputs: sum or last position
                ):
        """
        Forward pass of the model.

        Args:
            src1 (Tensor): First input sequence.
            src2 (Tensor): Second input sequence.
            src_mask (Tensor): Mask for the input sequence.
            src_key_padding_mask (Tensor): Padding mask for the input sequence.
            concat_type (str): Method to combine encoder outputs.

        Returns:
            Tensor: Model output.
        """
        src = torch.cat((src1, src2), dim=0)
        src_embed = self.src_token_embedding(src)  # Token embedding
        src_embed = self.pos_embedding(src_embed)  # Add positional encoding

        memory = self.encoder(src=src_embed,
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)

        x = memory.permute(1, 2, 0)  # Adjust dimensions for CNN input
        output = [conv(x) for conv in self.conv_list]  # Apply CNN layers
        output = torch.cat(output, dim=1)  # Concatenate CNN outputs
        output = output.view(-1, output.size(1))  # Flatten for linear layer
        output = self.dropout(output)  # Apply dropout

        return self.fc(F.gelu(output))  # Final linear layer with GELU activation


if __name__ == '__main__':
    # Example usage
    src_len = 7
    batch_size = 2
    dmodel = 100
    num_head = 4
    src = torch.tensor([[4, 3, 2, 6, 0, 0, 0],
                        [5, 7, 8, 2, 4, 0, 0]]).transpose(0, 1)  # Convert to [src_len, batch_size]
    src_key_padding_mask = torch.tensor([[True, True, True, True, False, False, False],
                                         [True, True, True, True, True, False, False]])
    model = KLModel_Plus(vocab_size=10, d_model=dmodel, nhead=num_head)
    logits = model(src, src_key_padding_mask=src_key_padding_mask)
    print(logits)
    print(logits.shape)