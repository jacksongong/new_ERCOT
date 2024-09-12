import os
import pickle
import warnings
import argparse
import pathlib

import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from torch.utils.data import Dataset, DataLoader
from configparser import ConfigParser
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from models.model_class_functions import gather_inputs, shift_sequence, manipulate_data
warnings.filterwarnings("ignore")


class MLP_Model(L.LightningModule):
    def __init__(self, attributes, x_train, y_train, x_validate, y_validate, x_test, y_test, learning_rate, batch_size, std_sys_lam, mean_sys_lam, accuracy_range):
        super(MLP_Model, self).__init__()
        self.std_sys_lam = std_sys_lam
        self.mean_sys_lam = mean_sys_lam
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
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.test_step_outputs = []
        self.x_test = x_test
        self.y_test = y_test
        self.accuracy_range = accuracy_range
        self.train_step_loss = []
        self.train_step_abosulte_error = []
        self.train_step_accuracy = []
        self.train_step_relative_error = []
        self.validation_step_loss = []
        self.validation_step_abosulte_error = []
        self.validation_step_accuracy = []
        self.validation_step_relative_error = []
        self.test_step_loss = []
        self.test_step_abosulte_error = []
        self.test_step_accuracy = []
        self.test_step_relative_error = []

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
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount/totalcount
        self.log('Train/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Train/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False,  logger=True)
        self.log('Train/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Train/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.train_step_outputs.append(y_predict_unnormalized)
        self.train_step_loss.append(loss)
        self.train_step_abosulte_error.append(torch.mean(absolute_error))
        self.train_step_relative_error.append(torch.mean(relative_error))
        self.train_step_accuracy.append(accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (
                    y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount / totalcount
        self.log('Validation/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.validation_step_outputs.append(y_predict_unnormalized)
        self.validation_step_loss.append(loss)
        self.validation_step_abosulte_error.append(torch.mean(absolute_error))
        self.validation_step_relative_error.append(torch.mean(relative_error))
        self.validation_step_accuracy.append(accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (
                    y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount / totalcount
        self.log('Test/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Test/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False, logger=True)
        self.log('Test/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Test/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.test_step_outputs.append(y_predict_unnormalized) #this code is not NESSESSARY
        self.test_step_loss.append(loss)
        self.test_step_abosulte_error.append(torch.mean(absolute_error))
        self.test_step_relative_error.append(torch.mean(relative_error))
        self.test_step_accuracy.append(accuracy)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        if self.train_step_outputs:
            avg_loss = sum(self.train_step_loss)/len(self.train_step_loss)
            avg_absolute_error = sum(self.train_step_abosulte_error)/len(self.train_step_abosulte_error)
            avg_relative_error = sum(self.train_step_relative_error)/len(self.train_step_relative_error)
            avg_accuracy = sum(self.train_step_accuracy)/len(self.train_step_accuracy)
            self.log('Train_Epoch/Loss_epoch', avg_loss)
            self.log('Train_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Train_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Train_Epoch/Accuracy_epoch', avg_accuracy)
            self.train_step_outputs.clear()
            self.train_step_loss.clear()
            self.train_step_relative_error.clear()
            self.train_step_abosulte_error.clear()
            self.train_step_accuracy.clear()

    def on_validation_epoch_end(self):  #sum(self.)/len(self.)
        if self.validation_step_outputs:
            avg_loss = sum(self.validation_step_loss)/len(self.validation_step_loss)
            avg_absolute_error = sum(self.validation_step_abosulte_error)/len(self.validation_step_abosulte_error)
            avg_relative_error = sum(self.validation_step_relative_error)/len(self.validation_step_relative_error)
            avg_accuracy = sum(self.validation_step_accuracy)/len(self.validation_step_accuracy)
            self.log('Validation_Epoch/Loss_epoch', avg_loss)
            self.log('Validation_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Validation_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Validation_Epoch/Accuracy_epoch', avg_accuracy)
            self.validation_step_outputs.clear()
            self.validation_step_loss.clear()
            self.validation_step_relative_error.clear()
            self.validation_step_abosulte_error.clear()
            self.validation_step_accuracy.clear()

    def on_test_epoch_end(self):  #this code is not NESSESSARY
        if self.test_step_outputs:
            avg_loss = sum(self.test_step_loss)/len(self.test_step_loss)
            avg_absolute_error = sum(self.test_step_abosulte_error)/len(self.test_step_abosulte_error)
            avg_relative_error = sum(self.test_step_relative_error)/len(self.test_step_relative_error)
            avg_accuracy = sum(self.test_step_accuracy)/len(self.test_step_accuracy)
            self.log('Test_Epoch/Loss_epoch', avg_loss)
            self.log('Test_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Test_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Test_Epoch/Accuracy_epoch', avg_accuracy)
            self.test_step_outputs.clear()
            self.test_step_loss.clear()
            self.test_step_relative_error.clear()
            self.test_step_abosulte_error.clear()
            self.test_step_accuracy.clear()

    def train_dataloader(self):
        x_train_array = self.x_train.values
        x_train_tensor = torch.tensor(x_train_array).float()
        y_train_array = self.y_train.values
        y_train_tensor = torch.tensor(y_train_array).float()
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        x_validate_array = self.x_validate.values
        x_validate_tensor = torch.tensor(x_validate_array).float()
        y_validate_array = self.y_validate.values
        y_validate_tensor = torch.tensor(y_validate_array).float()
        validate_dataset = torch.utils.data.TensorDataset(x_validate_tensor, y_validate_tensor)
        return DataLoader(validate_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        x_test_array = self.x_test.values
        x_test_tensor = torch.tensor(x_test_array).float()
        y_test_array = self.y_test.values
        y_test_tensor = torch.tensor(y_test_array).float()
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


class RNNModel(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim, x_train, y_train, x_validate, y_validate, x_test, y_test, learning_rate, batch_size, std_sys_lam, mean_sys_lam, accuracy_range):
        super(RNNModel, self).__init__()
        self.std_sys_lam = std_sys_lam
        self.mean_sys_lam = mean_sys_lam
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
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.x_test = x_test
        self.y_test = y_test
        self.accuracy_range = accuracy_range
        self.train_step_loss = []
        self.train_step_abosulte_error = []
        self.train_step_accuracy = []
        self.train_step_relative_error = []
        self.validation_step_loss = []
        self.validation_step_abosulte_error = []
        self.validation_step_accuracy = []
        self.validation_step_relative_error = []
        self.test_step_loss = []
        self.test_step_abosulte_error = []
        self.test_step_accuracy = []
        self.test_step_relative_error = []

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        while x.size(1) != h0.size(1):
            x = x[:, :h0.size(1), :]
        out, hn = self.rnn(x, h0)
        out = self.activation_function(self.fc1(out))
        out = self.activation_function(self.fc2(out))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (
                    y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount / totalcount
        self.log('Train/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Train/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False, logger=True)
        self.log('Train/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Train/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.train_step_outputs.append(y_predict_unnormalized)
        self.train_step_loss.append(loss)
        self.train_step_abosulte_error.append(torch.mean(absolute_error))
        self.train_step_relative_error.append(torch.mean(relative_error))
        self.train_step_accuracy.append(accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (
                y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount / totalcount
        self.log('Validation/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.validation_step_outputs.append(y_predict_unnormalized)
        self.validation_step_loss.append(loss)
        self.validation_step_abosulte_error.append(torch.mean(absolute_error))
        self.validation_step_relative_error.append(torch.mean(relative_error))
        self.validation_step_accuracy.append(accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (
                y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount / totalcount
        self.log('Test/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Test/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False, logger=True)
        self.log('Test/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Test/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.test_step_outputs.append(y_predict_unnormalized)  # this code is not NESSESSARY
        self.test_step_loss.append(loss)
        self.test_step_abosulte_error.append(torch.mean(absolute_error))
        self.test_step_relative_error.append(torch.mean(relative_error))
        self.test_step_accuracy.append(accuracy)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        if self.train_step_outputs:
            avg_loss = sum(self.train_step_loss) / len(self.train_step_loss)
            avg_absolute_error = sum(self.train_step_abosulte_error) / len(self.train_step_abosulte_error)
            avg_relative_error = sum(self.train_step_relative_error) / len(self.train_step_relative_error)
            avg_accuracy = sum(self.train_step_accuracy) / len(self.train_step_accuracy)
            self.log('Train_Epoch/Loss_epoch', avg_loss)
            self.log('Train_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Train_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Train_Epoch/Accuracy_epoch', avg_accuracy)
            self.train_step_outputs.clear()
            self.train_step_loss.clear()
            self.train_step_relative_error.clear()
            self.train_step_abosulte_error.clear()
            self.train_step_accuracy.clear()

    def on_validation_epoch_end(self):  # sum(self.)/len(self.)
        if self.validation_step_outputs:
            avg_loss = sum(self.validation_step_loss) / len(self.validation_step_loss)
            avg_absolute_error = sum(self.validation_step_abosulte_error) / len(self.validation_step_abosulte_error)
            avg_relative_error = sum(self.validation_step_relative_error) / len(self.validation_step_relative_error)
            avg_accuracy = sum(self.validation_step_accuracy) / len(self.validation_step_accuracy)
            self.log('Validation_Epoch/Loss_epoch', avg_loss)
            self.log('Validation_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Validation_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Validation_Epoch/Accuracy_epoch', avg_accuracy)
            self.validation_step_outputs.clear()
            self.validation_step_loss.clear()
            self.validation_step_relative_error.clear()
            self.validation_step_abosulte_error.clear()
            self.validation_step_accuracy.clear()

    def on_test_epoch_end(self):  # this code is not NESSESSARY
        if self.test_step_outputs:
            avg_loss = sum(self.test_step_loss) / len(self.test_step_loss)
            avg_absolute_error = sum(self.test_step_abosulte_error) / len(self.test_step_abosulte_error)
            avg_relative_error = sum(self.test_step_relative_error) / len(self.test_step_relative_error)
            avg_accuracy = sum(self.test_step_accuracy) / len(self.test_step_accuracy)
            self.log('Test_Epoch/Loss_epoch', avg_loss)
            self.log('Test_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Test_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Test_Epoch/Accuracy_epoch', avg_accuracy)
            self.test_step_outputs.clear()
            self.test_step_loss.clear()
            self.test_step_relative_error.clear()
            self.test_step_abosulte_error.clear()
            self.test_step_accuracy.clear()

    def train_dataloader(self):
        x_train_tensor_normalized_seq = shift_sequence(self.x_train, self.seq_dim)
        y_train_tensor_seq = shift_sequence(self.y_train, self.seq_dim)
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor_normalized_seq, y_train_tensor_seq)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        validation_features_normalized_seq = shift_sequence(self.x_validate, self.seq_dim)
        validation_sys_lam_tensor_normalized_seq = shift_sequence(self.y_validate, self.seq_dim)
        validation_dataset = torch.utils.data.TensorDataset(validation_features_normalized_seq,
                                                            validation_sys_lam_tensor_normalized_seq)
        return DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        test_features_normalized_seq = shift_sequence(x_test, seq_dim)
        test_sys_lam_tensor_normalized_seq = shift_sequence(y_test, seq_dim)
        test_dataset = torch.utils.data.TensorDataset(test_features_normalized_seq,
                                                      test_sys_lam_tensor_normalized_seq)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


class GRUModel(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim, x_train, y_train, x_validate, y_validate, x_test, y_test,
                 learning_rate, batch_size, std_sys_lam, mean_sys_lam, accuracy_range):
        super(GRUModel, self).__init__()
        self.std_sys_lam = std_sys_lam
        self.mean_sys_lam = mean_sys_lam
        self.x_train = x_train
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        self.activation_function = nn.PReLU()
        self.loss_function = nn.MSELoss()
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.x_test = x_test
        self.y_test = y_test
        self.accuracy_range = accuracy_range
        self.train_step_loss = []
        self.train_step_abosulte_error = []
        self.train_step_accuracy = []
        self.train_step_relative_error = []
        self.validation_step_loss = []
        self.validation_step_abosulte_error = []
        self.validation_step_accuracy = []
        self.validation_step_relative_error = []
        self.test_step_loss = []
        self.test_step_abosulte_error = []
        self.test_step_accuracy = []
        self.test_step_relative_error = []

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        while x.size(1) != h0.size(1):
            x = x[:, :h0.size(1), :]
        out, hn = self.gru(x, h0)
        out = self.activation_function(self.fc1(out))
        out = self.activation_function(self.fc2(out))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (
                    y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount / totalcount
        self.log('Train/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Train/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False, logger=True)
        self.log('Train/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Train/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.train_step_outputs.append(y_predict_unnormalized)
        self.train_step_loss.append(loss)
        self.train_step_abosulte_error.append(torch.mean(absolute_error))
        self.train_step_relative_error.append(torch.mean(relative_error))
        self.train_step_accuracy.append(accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (
                y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount / totalcount
        self.log('Validation/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.validation_step_outputs.append(y_predict_unnormalized)
        self.validation_step_loss.append(loss)
        self.validation_step_abosulte_error.append(torch.mean(absolute_error))
        self.validation_step_relative_error.append(torch.mean(relative_error))
        self.validation_step_accuracy.append(accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (
                y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount / totalcount
        self.log('Test/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Test/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False, logger=True)
        self.log('Test/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Test/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.test_step_outputs.append(y_predict_unnormalized)  # this code is not NESSESSARY
        self.test_step_loss.append(loss)
        self.test_step_abosulte_error.append(torch.mean(absolute_error))
        self.test_step_relative_error.append(torch.mean(relative_error))
        self.test_step_accuracy.append(accuracy)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        if self.train_step_outputs:
            avg_loss = sum(self.train_step_loss) / len(self.train_step_loss)
            avg_absolute_error = sum(self.train_step_abosulte_error) / len(self.train_step_abosulte_error)
            avg_relative_error = sum(self.train_step_relative_error) / len(self.train_step_relative_error)
            avg_accuracy = sum(self.train_step_accuracy) / len(self.train_step_accuracy)
            self.log('Train_Epoch/Loss_epoch', avg_loss)
            self.log('Train_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Train_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Train_Epoch/Accuracy_epoch', avg_accuracy)
            self.train_step_outputs.clear()
            self.train_step_loss.clear()
            self.train_step_relative_error.clear()
            self.train_step_abosulte_error.clear()
            self.train_step_accuracy.clear()

    def on_validation_epoch_end(self):  # sum(self.)/len(self.)
        if self.validation_step_outputs:
            avg_loss = sum(self.validation_step_loss) / len(self.validation_step_loss)
            avg_absolute_error = sum(self.validation_step_abosulte_error) / len(self.validation_step_abosulte_error)
            avg_relative_error = sum(self.validation_step_relative_error) / len(self.validation_step_relative_error)
            avg_accuracy = sum(self.validation_step_accuracy) / len(self.validation_step_accuracy)
            self.log('Validation_Epoch/Loss_epoch', avg_loss)
            self.log('Validation_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Validation_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Validation_Epoch/Accuracy_epoch', avg_accuracy)
            self.validation_step_outputs.clear()
            self.validation_step_loss.clear()
            self.validation_step_relative_error.clear()
            self.validation_step_abosulte_error.clear()
            self.validation_step_accuracy.clear()

    def on_test_epoch_end(self):  # this code is not NESSESSARY
        if self.test_step_outputs:
            avg_loss = sum(self.test_step_loss) / len(self.test_step_loss)
            avg_absolute_error = sum(self.test_step_abosulte_error) / len(self.test_step_abosulte_error)
            avg_relative_error = sum(self.test_step_relative_error) / len(self.test_step_relative_error)
            avg_accuracy = sum(self.test_step_accuracy) / len(self.test_step_accuracy)
            self.log('Test_Epoch/Loss_epoch', avg_loss)
            self.log('Test_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Test_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Test_Epoch/Accuracy_epoch', avg_accuracy)
            self.test_step_outputs.clear()
            self.test_step_loss.clear()
            self.test_step_relative_error.clear()
            self.test_step_abosulte_error.clear()
            self.test_step_accuracy.clear()

    def train_dataloader(self):
        x_train_tensor_normalized_seq = shift_sequence(self.x_train, self.seq_dim)
        y_train_tensor_seq = shift_sequence(self.y_train, self.seq_dim)
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor_normalized_seq, y_train_tensor_seq)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        validation_features_normalized_seq = shift_sequence(self.x_validate, self.seq_dim)
        validation_sys_lam_tensor_normalized_seq = shift_sequence(self.y_validate, self.seq_dim)
        validation_dataset = torch.utils.data.TensorDataset(validation_features_normalized_seq,
                                                            validation_sys_lam_tensor_normalized_seq)
        return DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        test_features_normalized_seq = shift_sequence(x_test, seq_dim)
        test_sys_lam_tensor_normalized_seq = shift_sequence(y_test, seq_dim)
        test_dataset = torch.utils.data.TensorDataset(test_features_normalized_seq,
                                                      test_sys_lam_tensor_normalized_seq)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

class LSTM_Model(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim, x_train, y_train, x_validate, y_validate, x_test, y_test,
                 learning_rate, batch_size, std_sys_lam, mean_sys_lam, accuracy_range):
        super(LSTM_Model, self).__init__()
        self.std_sys_lam = std_sys_lam
        self.mean_sys_lam = mean_sys_lam
        self.x_train = x_train
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        self.activation_function = nn.PReLU()
        self.loss_function = nn.MSELoss()
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.x_test = x_test
        self.y_test = y_test
        self.accuracy_range = accuracy_range
        self.train_step_loss = []
        self.train_step_abosulte_error = []
        self.train_step_accuracy = []
        self.train_step_relative_error = []
        self.validation_step_loss = []
        self.validation_step_abosulte_error = []
        self.validation_step_accuracy = []
        self.validation_step_relative_error = []
        self.test_step_loss = []
        self.test_step_abosulte_error = []
        self.test_step_accuracy = []
        self.test_step_relative_error = []

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        while x.size(1) != h0.size(1):
            x = x[:, :h0.size(1), :]
        while x.size(1) != c0.size(1):
            x = x[:, :c0.size(1), :]
        out, _ = self.lstm(x, (h0, c0))
        out = self.activation_function(self.fc1(out))
        out = self.activation_function(self.fc2(out))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (
                    y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount / totalcount
        self.log('Train/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Train/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False, logger=True)
        self.log('Train/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Train/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.train_step_outputs.append(y_predict_unnormalized)
        self.train_step_loss.append(loss)
        self.train_step_abosulte_error.append(torch.mean(absolute_error))
        self.train_step_relative_error.append(torch.mean(relative_error))
        self.train_step_accuracy.append(accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (
                y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount / totalcount
        self.log('Validation/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Validation/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.validation_step_outputs.append(y_predict_unnormalized)
        self.validation_step_loss.append(loss)
        self.validation_step_abosulte_error.append(torch.mean(absolute_error))
        self.validation_step_relative_error.append(torch.mean(relative_error))
        self.validation_step_accuracy.append(accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        y_predict_unnormalized = y_hat * self.std_sys_lam + self.mean_sys_lam
        y_predict_unnormalized = abs(y_predict_unnormalized)
        y_unnormalized = y * self.std_sys_lam + self.mean_sys_lam
        y_unnormalized = abs(y_unnormalized)
        absolute_error = torch.abs(y_predict_unnormalized - y_unnormalized)
        percent_within_lower_bound = (accuracy_range) * y_unnormalized
        percent_within_upper_bound = (2 - accuracy_range) * y_unnormalized
        within_bounds = (y_predict_unnormalized >= percent_within_lower_bound) & (
                y_predict_unnormalized <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(y_unnormalized)
        relative_error = absolute_error / y_unnormalized
        accuracy = workingcount / totalcount
        self.log('Test/Loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('Test/Absolute_Error', torch.mean(absolute_error), on_step=True, on_epoch=False, logger=True)
        self.log('Test/Relative_Error', torch.mean(relative_error), on_step=True, on_epoch=False, logger=True)
        self.log('Test/Accuracy', accuracy, on_step=True, on_epoch=False, logger=True)
        self.test_step_outputs.append(y_predict_unnormalized)  # this code is not NESSESSARY
        self.test_step_loss.append(loss)
        self.test_step_abosulte_error.append(torch.mean(absolute_error))
        self.test_step_relative_error.append(torch.mean(relative_error))
        self.test_step_accuracy.append(accuracy)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        if self.train_step_outputs:
            avg_loss = sum(self.train_step_loss) / len(self.train_step_loss)
            avg_absolute_error = sum(self.train_step_abosulte_error) / len(self.train_step_abosulte_error)
            avg_relative_error = sum(self.train_step_relative_error) / len(self.train_step_relative_error)
            avg_accuracy = sum(self.train_step_accuracy) / len(self.train_step_accuracy)
            self.log('Train_Epoch/Loss_epoch', avg_loss)
            self.log('Train_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Train_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Train_Epoch/Accuracy_epoch', avg_accuracy)
            self.train_step_outputs.clear()
            self.train_step_loss.clear()
            self.train_step_relative_error.clear()
            self.train_step_abosulte_error.clear()
            self.train_step_accuracy.clear()

    def on_validation_epoch_end(self):  # sum(self.)/len(self.)
        if self.validation_step_outputs:
            avg_loss = sum(self.validation_step_loss) / len(self.validation_step_loss)
            avg_absolute_error = sum(self.validation_step_abosulte_error) / len(self.validation_step_abosulte_error)
            avg_relative_error = sum(self.validation_step_relative_error) / len(self.validation_step_relative_error)
            avg_accuracy = sum(self.validation_step_accuracy) / len(self.validation_step_accuracy)
            self.log('Validation_Epoch/Loss_epoch', avg_loss)
            self.log('Validation_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Validation_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Validation_Epoch/Accuracy_epoch', avg_accuracy)
            self.validation_step_outputs.clear()
            self.validation_step_loss.clear()
            self.validation_step_relative_error.clear()
            self.validation_step_abosulte_error.clear()
            self.validation_step_accuracy.clear()

    def on_test_epoch_end(self):  # this code is not NESSESSARY
        if self.test_step_outputs:
            avg_loss = sum(self.test_step_loss) / len(self.test_step_loss)
            avg_absolute_error = sum(self.test_step_abosulte_error) / len(self.test_step_abosulte_error)
            avg_relative_error = sum(self.test_step_relative_error) / len(self.test_step_relative_error)
            avg_accuracy = sum(self.test_step_accuracy) / len(self.test_step_accuracy)
            self.log('Test_Epoch/Loss_epoch', avg_loss)
            self.log('Test_Epoch/Absolute_Error_epoch', avg_absolute_error)
            self.log('Test_Epoch/Relative_Error_epoch', avg_relative_error)
            self.log('Test_Epoch/Accuracy_epoch', avg_accuracy)
            self.test_step_outputs.clear()
            self.test_step_loss.clear()
            self.test_step_relative_error.clear()
            self.test_step_abosulte_error.clear()
            self.test_step_accuracy.clear()

    def train_dataloader(self):
        x_train_tensor_normalized_seq = shift_sequence(self.x_train, self.seq_dim)
        y_train_tensor_seq = shift_sequence(self.y_train, self.seq_dim)
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor_normalized_seq, y_train_tensor_seq)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        validation_features_normalized_seq = shift_sequence(self.x_validate, self.seq_dim)
        validation_sys_lam_tensor_normalized_seq = shift_sequence(self.y_validate, self.seq_dim)
        validation_dataset = torch.utils.data.TensorDataset(validation_features_normalized_seq,
                                                            validation_sys_lam_tensor_normalized_seq)
        return DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        test_features_normalized_seq = shift_sequence(x_test, seq_dim)
        test_sys_lam_tensor_normalized_seq = shift_sequence(y_test, seq_dim)
        test_dataset = torch.utils.data.TensorDataset(test_features_normalized_seq,
                                                      test_sys_lam_tensor_normalized_seq)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


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
    # exp_number = args.exp_number
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
    exp_number = config.getint('exp_number')
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
    accuracy_range = config.getfloat('accuracy_range')
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device\n")
    attributes = gather_inputs(data_path)
    print(model_type)
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
    x_train_tensor, y_train_tensor, x_validate_tensor, y_validate_tensor, x_test, y_test, std_sys_lam, mean_sys_lam = manipulate_data(data_path, validate_size,
                                                                                           test_size, random_state,
                                                                                           batch_size, model_class,
                                                                                           model_params, device,
                                                                                           seq_dim)
    if not model_params:
        model = model_class(attributes, x_train_tensor, y_train_tensor, x_validate_tensor, y_validate_tensor, x_test, y_test, learning_rate, batch_size, std_sys_lam, mean_sys_lam, accuracy_range)
    else:
        model = model_class(attributes, hidden_dim, layer_dim, seq_dim, output_dim, x_train_tensor, y_train_tensor, x_validate_tensor, y_validate_tensor, x_test, y_test,
                            learning_rate, batch_size, std_sys_lam, mean_sys_lam, accuracy_range)
    Path_model = os.path.join(model_path, f"exp{exp_number}_Model_Path")
    torch.save(model.state_dict(), Path_model)
    loaded_model = model.load_state_dict(torch.load(Path_model))
    trainer = L.Trainer(max_epochs=number_epochs, accelerator='gpu', devices=1, num_sanity_val_steps=0)
    trainer.fit(model)
    if not model_params:
        x_test_array = x_test.values
        x_test_tensor = torch.tensor(x_test_array).float()
        y_test_array = y_test.values
        y_test_tensor = torch.tensor(y_test_array).float()
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        test_features_normalized_seq = shift_sequence(x_test, seq_dim)
        test_sys_lam_tensor_normalized_seq = shift_sequence(y_test, seq_dim)
        test_dataset = torch.utils.data.TensorDataset(test_features_normalized_seq,
                                                      test_sys_lam_tensor_normalized_seq)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    trainer.test(dataloaders = test_dataloader)








