import os
import pickle
import warnings
import argparse
import pathlib

import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from lightning.pytorch.demos import SequenceSampler
import lightning as L
from torch.utils.data import Dataset, DataLoader
from configparser import ConfigParser
import torch
import torch.nn as nn
import pandas as pd

from models.model_class_functions import gather_inputs, shift_sequence, manipulate_data
warnings.filterwarnings("ignore")


class Proccess_Model(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch):
        # Send the batch through the model and calculate the loss
        # The Trainer will run .backward(), optimizer.step(), .zero_grad(), etc. for you
        loss = self.model(batch).sum()
        return loss

    def configure_optimizers(self):
        # Choose an optimizer or implement your own.
        return torch.optim.Adam(self.model.parameters())


class LanguageModel(L.LightningModule):
    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type
        self.hidden = None

    def on_train_epoch_end(self):
        self.hidden = None

    def training_step(self, batch, batch_idx):
        input, target = batch
        if self.hidden is None:
            self.hidden = self.model_type.init_hidden(input.size(0))
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        output, self.hidden = self.model_type(input, self.hidden)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=20.0)





class MLP_Model(L.LightningModule):
    def __init__(self, attributes, x_train, y_train, x_validate, y_validate, learning_rate, batch_size):
        super(MLP_Model, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.batch_size = batch_size
        self.layer1 = nn.Linear(attributes, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer4 = nn.Linear(1024, 1024)
        self.layer5 = nn.Linear(1024, 1024)
        self.layer6 = nn.Linear(1024, 1)
        self.activation_function = nn.PReLU()
        self.learning_rate = learning_rate
        self.loss_function = nn.MSELoss()

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)
        nn.init.xavier_uniform_(self.layer5.weight)
        nn.init.xavier_uniform_(self.layer6.weight)

    def forward(self, x):
        x = self.activation_function(self.layer1(x))
        x = self.activation_function(self.layer2(x))
        x = self.activation_function(self.layer3(x))
        x = self.activation_function(self.layer4(x))
        x = self.activation_function(self.layer5(x))
        x = self.layer6(x)
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        print(f"Training Loss: {loss}\n")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        print(f"Validation Loss: {loss}\n")
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        x_train_array = self.x_train.values
        x_train_tensor = torch.tensor(x_train_array).float
        y_train_array = self.y_train.values
        y_train_tensor = torch.tensor(y_train_array).float
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        x_validate_array = self.x_validate.values
        x_validate_tensor = torch.tensor(x_validate_array).float
        y_validate_array = self.y_validate.values
        y_validate_tensor = torch.tensor(y_validate_array).float
        validate_dataset = torch.utils.data.TensorDataset(x_validate_tensor, y_validate_tensor)
        return DataLoader(validate_dataset, batch_size=self.batch_size, shuffle=True)

class RNNModel(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim, x_train, y_train, x_validate, y_validate, learning_rate, batch_size):
        super(RNNModel, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc1 = nn.Linear(hidden_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        self.activation_function = nn.PReLU()
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        print(x.shape)
        print(h0.shape)
        out, hn = self.rnn(x, h0)
        out = self.activation_function(self.fc1(out))
        out = self.activation_function(self.fc2(out))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        print(y_hat)
        print(y)
        loss = self.loss_function(y_hat, y)
        absolute_error = torch.abs(y_hat - y)
        relative_error = absolute_error / y
        accuracy = torch.mean((1 - relative_error)).item()
        self.log('train_loss', loss)
        self.log('train_absolute_error', torch.mean(absolute_error))
        self.log('train_relative_error', torch.mean(relative_error))
        self.log('train_accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        absolute_error = torch.abs(y_hat - y)
        relative_error = absolute_error / y
        accuracy = torch.mean((1 - relative_error)).item()
        self.log('val_loss', loss)
        self.log('val_absolute_error', torch.mean(absolute_error))
        self.log('val_relative_error', torch.mean(relative_error))
        self.log('val_accuracy', accuracy)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        x_train_tensor_normalized_seq = shift_sequence(self.x_train, self.seq_dim)
        y_train_tensor_seq = shift_sequence(self.y_train, self.seq_dim)
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor_normalized_seq, y_train_tensor_seq)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        validation_features_normalized_seq = shift_sequence(self.x_validate, self.seq_dim)
        validation_sys_lam_tensor_normalized_seq = shift_sequence(self.y_validate, self.seq_dim)
        test_dataset = torch.utils.data.TensorDataset(validation_features_normalized_seq,
                                                      validation_sys_lam_tensor_normalized_seq)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


class GRUModel(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim, x_train, y_train, x_validate, y_validate,
                 learning_rate, batch_size):
        super(GRUModel, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc1 = nn.Linear(hidden_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        self.activation_function = nn.PReLU()
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.gru(x, h0)
        out = self.activation_function(self.fc1(out))
        out = self.activation_function(self.fc2(out))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        absolute_error = torch.abs(y_hat - y)
        relative_error = absolute_error / y
        accuracy = torch.mean((1 - relative_error)).item()
        self.log('train_loss', loss)
        self.log('train_absolute_error', torch.mean(absolute_error))
        self.log('train_relative_error', torch.mean(relative_error))
        self.log('train_accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        absolute_error = torch.abs(y_hat - y)
        relative_error = absolute_error / y
        accuracy = torch.mean((1 - relative_error)).item()
        self.log('val_loss', loss)
        self.log('val_absolute_error', torch.mean(absolute_error))
        self.log('val_relative_error', torch.mean(relative_error))
        self.log('val_accuracy', accuracy)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        x_train_tensor_normalized_seq = shift_sequence(self.x_train, self.seq_dim)
        y_train_tensor_seq = shift_sequence(self.y_train, self.seq_dim)
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor_normalized_seq, y_train_tensor_seq)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        validation_features_normalized_seq = shift_sequence(self.x_validate, self.seq_dim)
        validation_sys_lam_tensor_normalized_seq = shift_sequence(self.y_validate, self.seq_dim)
        test_dataset = torch.utils.data.TensorDataset(validation_features_normalized_seq,
                                                      validation_sys_lam_tensor_normalized_seq)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


class LSTM_Model(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim, x_train, y_train, x_validate, y_validate,
                 learning_rate, batch_size):
        super(LSTM_Model, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc1 = nn.Linear(hidden_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        self.activation_function = nn.PReLU()
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.lstm(x, h0)
        out = self.activation_function(self.fc1(out))
        out = self.activation_function(self.fc2(out))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        absolute_error = torch.abs(y_hat - y)
        relative_error = absolute_error / y
        accuracy = torch.mean((1 - relative_error)).item()
        self.log('train_loss', loss)
        self.log('train_absolute_error', torch.mean(absolute_error))
        self.log('train_relative_error', torch.mean(relative_error))
        self.log('train_accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        absolute_error = torch.abs(y_hat - y)
        relative_error = absolute_error / y
        accuracy = torch.mean((1 - relative_error)).item()
        self.log('val_loss', loss)
        self.log('val_absolute_error', torch.mean(absolute_error))
        self.log('val_relative_error', torch.mean(relative_error))
        self.log('val_accuracy', accuracy)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        x_train_tensor_normalized_seq = shift_sequence(self.x_train, self.seq_dim)
        y_train_tensor_seq = shift_sequence(self.y_train, self.seq_dim)
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor_normalized_seq, y_train_tensor_seq)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        validation_features_normalized_seq = shift_sequence(self.x_validate, self.seq_dim)
        validation_sys_lam_tensor_normalized_seq = shift_sequence(self.y_validate, self.seq_dim)
        test_dataset = torch.utils.data.TensorDataset(validation_features_normalized_seq,
                                                      validation_sys_lam_tensor_normalized_seq)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

# STEP 2: RUN THE TRAINER
if __name__ == "__main__":
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
        help="name of config file"
    )

    parser.add_argument(
        "--number_epochs",
        type=int,
        required=True,
        help="name of config file"
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
    if (model_type == "Linear_L1" or model_type == "Linear_L2"):
        print(f"Using {model_type} Regression")
    elif model_type=="LSTM_Model" or model_type=="GRUModel" or model_type=="RNNModel" or model_type=="MLP_Model":
        model_class = globals()[model_type]
    else:
        raise ValueError("The wrong class name used in config file")

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
    learning_rate = 1e-6
    if not model_params:
        train_dataset, test_dataset = (
            manipulate_data(data_path, validate_size, test_size, random_state, batch_size, model_class, model_params,
                            device, seq_dim))
    else:
        x_train_tensor, y_train_tensor, x_validate_tensor, y_validate_tensor, train_dataset, test_dataset = (
            manipulate_data(data_path, validate_size, test_size, random_state, batch_size, model_class, model_params,
                            device, seq_dim))


    # if not model_params:
    #     model = model_class(attributes, x_train_tensor, y_train_tensor, x_validate_tensor, y_validate_tensor, learning_rate, batch_size)
    # else:
    #     model = model_class(attributes, hidden_dim, layer_dim, seq_dim, output_dim, x_train_tensor, y_train_tensor, x_validate_tensor, y_validate_tensor,
    #              learning_rate, batch_size)

    print(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_sampler=SequenceSampler(train_dataset, batch_size=20))
    model = LanguageModel(model_class)
    pl_module = Proccess_Model(model)
    trainer = L.Trainer(gradient_clip_val=0.25, max_epochs=20)
    trainer.fit(pl_module, train_dataloader)




    #validation_loaded_data
