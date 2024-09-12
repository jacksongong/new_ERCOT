import argparse
import pathlib
import os
from configparser import ConfigParser
import pickle
import warnings

import torch
import torch.nn as nn
import pandas as pd

import models.model_class_functions
from models.model_class_functions import gather_inputs, run_model, Linear_Regression_L1, Linear_Regression_L2
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Enter Directory to the Config File, Include the path to the created ERCOT data folder from clean_data argparse',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="path to the configuration file"
    )

    parser.add_argument(
        "--name_config",
        type=str,
        required=True,
        help="name of config file"
    )
    parser.add_argument(
        "--exp_number",
        type=int,
        required=True,
        help="experiment count (ID)"
    )

    parser.add_argument(
        "--number_epochs",
        type=int,
        required=True,
        help="number_epochs to train and analyize validation"
    )
    args = parser.parse_args()
    name_config = args.name_config
    exp_number = args.exp_number
    number_epochs = args.number_epochs

    def parse_config(config_file, name_config):
        config = ConfigParser()
        PATH = os.path.join(config_file, name_config)
        config.read(PATH)
        return config['DEFAULT']
    config = parse_config(args.config, name_config)
    data_path = pathlib.Path(config.get('data_path'))
    name_csv = config.get('name_csv')
    model_path = config.get('model_path')
    model_type = config.get('model_type')
    # exp_number = config.getint('exp_number')
    # number_epochs = config.getint('number_epochs')
    loss_function = config.get('loss_function')
    batch_size = config.getint('batch_size')
    seq_dim = config.getint('seq_dim')
    random_state = config.getint('random_state')
    test_size = config.getfloat('test_size')
    validate_size = config.getfloat('validate_size')
    confidence = config.getfloat('confidence')
    hidden_dim = config.getint('hidden_dim')
    layer_dim = config.getint('layer_dim')
    output_dim = config.getint('output_dim')
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device\n")
    attributes = gather_inputs(data_path)
    if (model_type == "Linear_Regessor_Pytorch" or model_type == "MLP_Model"):
        model_params = {
        }
    else:
        model_params = {
            'input_dim': attributes,
            'hidden_dim': hidden_dim,
            'layer_dim': layer_dim,
            'seq_dim': seq_dim,
            'output_dim': output_dim,
        }
    if (model_type == "Linear_L1" or model_type == "Linear_L2"):
        print(f"Using {model_type} Regression")
    else:
        model_class = getattr(models.model_class_functions, model_type)
    loss_function = getattr(nn, loss_function)()
    if model_type == "Linear_L1":
        x_test, y_test, x_test_unnormalized, y_test_unnormalized = Linear_Regression_L1(data_path, test_size, random_state, validate_size)
    elif model_type == "Linear_L2":
        x_test, y_test, x_test_unnormalized, y_test_unnormalized = Linear_Regression_L2(data_path, test_size, random_state, validate_size)
    else:
        x_test, y_test, x_test_unnormalized, y_test_unnormalized, x_train_tensor, x_validate_tensor, y_train_tensor, y_validate_tensor, = run_model(data_path, model_path, exp_number, number_epochs, validate_size, test_size, random_state, batch_size, confidence, model_class, loss_function, model_params, device, seq_dim)
    print("---------------------------------------------\nTesting Model:")
    Path_model = os.path.join(model_path, f"exp{exp_number}_Model_Path")
    if (model_type == "MLP_Model"):
        loaded_model = model_class(attributes)
    else:
        loaded_model = model_class(**model_params)
    loaded_model.load_state_dict(torch.load(Path_model))
    print(loaded_model)
    print("Now we do testing on the loaded_model")
    test_concated_normalized_data = pd.concat([x_test, y_test], axis=1)
    core_directory = pathlib.Path(".")
    csv_folder_path = core_directory / 'csv_test_data'
    pickle_folder_path = core_directory / 'pickle_test_data'
    csv_folder_path.mkdir(parents=True, exist_ok=True)
    pickle_folder_path.mkdir(parents=True, exist_ok=True)
    pickle_file_path_normalized = os.path.join(pickle_folder_path, f"exp{exp_number}_Normlized_Testing_Data.pkl")
    pickle_file_path_un_normalized = os.path.join(pickle_folder_path, f"exp{exp_number}_un_Normlized_Testing_Data.pkl")
    csv_file_path_normalized = os.path.join(csv_folder_path, f"exp{exp_number}_Normlized_Testing_Data.csv")
    csv_file_path_un_normalized = os.path.join(csv_folder_path, f"exp{exp_number}_un_Normlized_Testing_Data.csv")
    test_concated_normalized_data.to_csv(csv_file_path_normalized, index=False)
    with open(pickle_file_path_normalized, 'wb') as file:
        pickle.dump(test_concated_normalized_data, file)
    test_concated_un_normalized_data = pd.concat([x_test_unnormalized, y_test_unnormalized], axis=1)
    test_concated_un_normalized_data.to_csv(csv_file_path_un_normalized, index=False)
    with open(pickle_file_path_un_normalized, 'wb') as file:
        pickle.dump(test_concated_un_normalized_data, file)