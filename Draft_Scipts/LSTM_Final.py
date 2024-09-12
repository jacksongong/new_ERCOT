import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse
import pathlib
from torchvision.utils import make_grid
import datetime
import os
import time
import warnings
from Model_Class_Functions import LSTM_Model,  run_model, gather_inputs
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Final_Data_Proccessor INPUTS: Please run the python script from the parent directory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        required=True,
        help="path to the stored train and test csv files"
    )

    parser.add_argument(
        "--name_csv",
        type=str,
        required=True,
        help="name of csv file with all the data"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path to saved model after training"
    )

    parser.add_argument(
        "--exp_number",
        type=int,
        required=True,
        help="experiment number under the log folder"
    )

    parser.add_argument(
        "--number_epochs",
        type=int,
        required=True,
        help="number of trained epochs"
    )



    args = parser.parse_args()
    data_path = args.data_path
    name_csv = args.name_csv
    model_path = args.model_path
    exp_number = args.exp_number
    number_epochs = args.number_epochs
    confidence = 0.80
    test_size = 0.155
    random_state= 42
    batch_size = 64

    loss_function = nn.MSELoss()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device\n")
    attributes = gather_inputs(data_path, test_size, random_state)
    seq_dim = 7
    model_params = {
        'input_dim': attributes,
        'hidden_dim': 64,
        'layer_dim': 2,
        'seq_dim': 7,
        'output_dim': 1,
        'device': device
    }

    run_model(data_path, model_path, exp_number, number_epochs, test_size, random_state, batch_size, confidence, LSTM_Model, loss_function, model_params, device, seq_dim)