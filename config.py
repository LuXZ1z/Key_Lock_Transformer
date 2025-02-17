import os
import torch


class Config:
    """
    Configuration class for model and training settings.
    """
    def __init__(self):
        # Project and dataset directories
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.project_dir, 'input')
        self.train_corpus_file_paths = os.path.join(self.dataset_dir, 'train_set.csv')
        self.test_corpus_file_paths = os.path.join(self.dataset_dir, 'test_set.csv')
        self.dataset_file_paths = os.path.join(self.dataset_dir, 'original_data/sep_depth_32x12000_median_sample.csv')

        # Data processing settings
        self.min_freq = 1
        self.max_sen_len1 = 8
        self.max_sen_len2 = 8

        # Model architecture settings
        self.batch_size = 256
        self.d_model = 100
        self.num_head = 5
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 64
        self.dim_output = 64
        self.output = 1
        self.dropout = 0.1
        self.k_mer1 = 3
        self.k_mer2 = 3
        self.concat_type = 'sum'

        # Training settings
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 1e-8
        self.weight_decay = 1e-5
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = 200
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.model_save_per_epoch = 1
        self.early_stopping_rounds = 50

        # Loss function settings
        self.init_noise_sigma = 8.0
        self.sigma_lr = 1e-2
        self.pretrained_emb = True

        # CNN settings
        self.cnn_channels = 256
        self.kernel_sizes = [10] * 15  # List of kernel sizes for CNN layers

        # Ensure the model save directory exists
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)


if __name__ == '__main__':
    config = Config()
    print("Device:", config.device)
    print("Project Directory:", config.project_dir)
    print("Training Corpus File Path:", config.train_corpus_file_paths)