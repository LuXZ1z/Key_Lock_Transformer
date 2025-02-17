import os

import matplotlib

from analysis_data import add_key_info, process_32x12000, evaluate_model
from config import Config
from infer import run_infer
from train import train_model
from utils import get_timestamp

matplotlib.use('Agg')  # Use Agg backend for rendering
timestamp = get_timestamp()

if __name__ == '__main__':
    """
    Main script to train a model, perform inference, and analyze results.

    This script orchestrates the entire workflow from training a model to performing inference and analyzing the results.
    It uses a timestamp to uniquely identify each run and organize output files and directories.
    """
    # Initialize configuration
    config = Config()

    # Get current timestamp for model identification
    timestamp = get_timestamp()

    # Define file paths and model directory
    file_path = r"./input/original_data/sep_depth_32x12000_median_sample.csv"  # Dataset for inference
    model_name = timestamp
    model_dir = os.path.join(config.model_save_dir, model_name)
    output_file_name = f'32x12000_{model_name}.csv'
    original_data_path = r"./input/original_data/original_data.csv"  # Original dataset with additional information

    # Train the model
    train_model(config, timestamp)

    # Run inference using the trained model
    run_infer(config, file_path, output_file_name, model_dir)