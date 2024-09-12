import torch
import argparse
import pathlib
import os
import pickle
from Model_Class_Functions import MLP_Model, RNNModel, LSTM_Model, GRUModel
import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    class_model = RNNModel  #change this based on the model desired

    device = torch.device("cuda")

    parser = argparse.ArgumentParser(
        description='Load Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path to saved model after training"
    )

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
        "--exp_number",
        type=int,
        required=True,
        help="experiment number under the log folder"
    )

    args = parser.parse_args()
    model_path = args.model_path
    exp_number = args.exp_number
    data_path = args.data_path
    name_csv = args.name_csv


    PATH = os.path.join(model_path, f"exp{exp_number}_Model_Path")
    concat_data, before_sales_column, attributes = process_data(data_path, name_csv)


    model_params = {
        'input_dim': attributes,
        'hidden_dim': 64,
        'layer_dim': 2,
        'seq_dim': 7,
        'output_dim': 1,
        'device': device
    }

    if not model_params:
        model = class_model(device, attributes).to(device)
    else:
        model = class_model(**model_params).to(device)

    model.load_state_dict(torch.load(PATH))

    model.eval()
    print(model)
    print(model.state_dict())