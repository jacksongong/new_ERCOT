import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse
import pathlib
from torchvision.utils import make_grid
import datetime
import os
#import tensorflow as tf
import time
import pickle


def process_data(data_path, name_csv):
    df_function = os.path.join(data_path, name_csv)

    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 20)

    df = pd.read_csv(df_function)

    common_time = 'Hour_Ending'
    common_delivery = 'DeliveryDate'
    actual_desired_lambda = 'SystemLambda_ActualValues'

    before_sales_column = df[actual_desired_lambda]
    df['DateTime'] = pd.to_datetime(df['DeliveryDate'])
    df['weekday'] = df['DateTime'].dt.weekday
    ohe_weekdays = pd.get_dummies(df['weekday'], prefix='weekday')
    ohe_weekdays = ohe_weekdays.astype(bool).astype(int)
    df['month'] = df['DateTime'].dt.month
    ohe_months = pd.get_dummies(df['month'], prefix='month')
    ohe_months = ohe_months.astype(bool).astype(int)
    df['hour'] = df['Hour_Ending'].str.split(':').str[0].astype(int)
    ohe_hours = pd.get_dummies(df['hour'], prefix='hour')
    ohe_hours = ohe_hours.astype(bool).astype(int)
    train_drop = df.drop(columns=['weekday', 'month', 'hour', common_delivery, common_time, actual_desired_lambda, 'DateTime'])

    train_drop = train_drop.iloc[:, 1:]
    before_numbers_train = train_drop.select_dtypes(exclude=['object'])
    before_numbers_train.fillna(0, inplace=True)

    square_df = pd.DataFrame()

    for column in before_numbers_train.columns:
        square_df[f"{column}_squared"] = before_numbers_train[column] ** 2
    for column in before_numbers_train.columns:
        square_df[f"{column}_cube"] = before_numbers_train[column] ** 3
    for column in before_numbers_train.columns:
        square_df[f"{column}_quartic"] = before_numbers_train[column] ** 4
    for column in before_numbers_train.columns:
        square_df[f"{column}_quintic"] = before_numbers_train[column] ** 5
    for column in before_numbers_train.columns:
        square_df[f"{column}_sqrt"] = before_numbers_train[column] ** 0.5
    for column in before_numbers_train.columns:
        square_df[f"{column}_log"] = np.log10(before_numbers_train[column])

    before_numbers_train_concat_data = pd.concat([before_numbers_train, square_df], axis=1)
    before_split_train_manipulated = before_numbers_train_concat_data.apply(lambda x: (x - x.mean()) / (x.std()))

    concat_data = pd.concat([ohe_weekdays, ohe_months, ohe_hours, before_split_train_manipulated], axis=1)
    concat_data = concat_data.select_dtypes(exclude=['object'])
    concat_data.fillna(0, inplace=True)
    attributes = concat_data.shape[1]

    #the below code only has to be done ONCE
    csv_file_path = os.path.join(data_path, "Hot_Encoded_Data.csv")
    pickle_path_updated = os.path.join(data_path, "Hot_Encode.pkl")
    concat_data.to_csv(csv_file_path, index=True)
    with open(pickle_path_updated, 'wb') as file:
        pickle.dump(concat_data, file)

    return concat_data, before_sales_column, attributes


def run_model(data_path, name_csv, model_path, exp_number, number_epochs, test_size, random_state, batch_size, confidence, class_model, loss_function, model_params, device):
    concat_data, before_sales_column, attributes = process_data(data_path, name_csv)
    stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size, random_state=random_state)
    sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size, shuffle=False, random_state=random_state)

    input = stat_model.shape[0]
    attibutes = stat_model.shape[1]

    train_features = torch.tensor(stat_model.values).float()
    validation_features = torch.tensor(stat_model_validation.values).float()

    sales_column_tensor = torch.tensor(sales_column.values).float()

    validation_sales_tensor = torch.tensor(validation_sales.values).float()
    mean_validation = validation_sales_tensor.mean()
    std_validation = validation_sales_tensor.std()
    validation_normalized_sales = (validation_sales_tensor - mean_validation) / std_validation

    validation_sales = validation_sales.tolist()

    mean_sales = sales_column_tensor.mean()
    std_sales = sales_column_tensor.std()
    normalized_sales = (sales_column_tensor - mean_sales) / std_sales

    normalized_tensor_sales = torch.tensor(normalized_sales).float()

    train_dataset_custom_dataset = CustomTensorDataset(train_features, normalized_tensor_sales, device)
    validation_dataset_custom_dataset = CustomTensorDataset(validation_features, validation_normalized_sales, device)

    train_loaded_data = DataLoader(train_dataset_custom_dataset, batch_size=batch_size, shuffle=True)
    validation_loaded_data = DataLoader(validation_dataset_custom_dataset, batch_size=batch_size, shuffle=False)

    if not model_params:
        model = class_model(device, attibutes).to(device)
    else:
        model = class_model(**model_params).to(device)

    writer = SummaryWriter(f"log/exp{exp_number}")
    global_step_list = [1]

    epochs = number_epochs
    global_step_loss = [1]
    for i in range(epochs):
        global_step_loss.append(i + 1)
    epoch_count = 0
    shown_data = 20
    epoch_list = [i for i in range(1, epochs + 1)]
    predictions_validation = []
    loss_set_iteration = []
    percent_set_iteration = []
    average_percent = []
    average_abs = []
    absolute_percent = []
    weight_decay = 1e-4
    total_batch_epoch_accuracy = []
    total_batch_epoch_loss = []

    if epoch_count <= 5:
        learning_rate = 1e-5
    else:
        learning_rate = pow(1, -epoch_count)

    for k in range(epochs):
        print(f"-------------------------------------\nTraining the {k + 1} epoch:\n")
        start_time = time.time()

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_batch_loss, batch_epoch_accuracy, batch_epoch_loss, global_step = train_all(model, optimizer, train_loaded_data, global_step_list[k], k, epoch_list, global_step_loss[k], confidence, loss_function, device, exp_number, std_sales, mean_sales, writer)


        global_step_list.append(global_step)
        total_batch_epoch_accuracy.append(sum(batch_epoch_accuracy) / len(batch_epoch_accuracy))
        total_batch_epoch_loss.append(sum(batch_epoch_loss) / len(batch_epoch_loss))

        predictions_validation_iteration = []
        model.eval()

        with torch.no_grad():
            for features in validation_loaded_data:
                outputs = model(features[0])
                outputs = outputs.to(device)
                final_value = ((outputs * std_validation) + mean_validation)
                for i in range(len(final_value)):
                    predictions_validation_iteration.append(final_value[i].item())
                predictions_validation.append(predictions_validation_iteration)

        tensor_validaion_iteration = torch.tensor(predictions_validation_iteration).float()
        tensor_validation_sales = torch.tensor(validation_sales).float()

        mean_tensor_validaion_iteration = tensor_validaion_iteration.mean()
        std_tensor_validaion_iteration = tensor_validaion_iteration.std()
        mean_tensor_validation_sales = tensor_validation_sales.mean()
        std_tensor_validation_sales = tensor_validation_sales.std()

        normalized_tensor_validaion_iteration = (tensor_validaion_iteration - mean_tensor_validation_sales) / std_tensor_validation_sales
        normalized_tensor_validation_sales = (tensor_validation_sales - mean_tensor_validation_sales) / std_tensor_validation_sales

        loss_validation = loss_function(normalized_tensor_validaion_iteration, normalized_tensor_validation_sales)
        loss_set_iteration.append(loss_validation.item())

        for i in range(len(predictions_validation_iteration)):
            totalcount = 0
            workingcount = 0
            percent_within_lower_bound = (confidence) * validation_sales[i]
            percent_within_upper_bound = (2 - confidence) * validation_sales[i]

            workingcount += percent_within_lower_bound <= predictions_validation_iteration[i] <= percent_within_upper_bound
            totalcount += 1
            differance = predictions_validation_iteration[i] - validation_sales[i]
            percent_set_iteration.append(abs(100 * (differance / validation_sales[i])))
            absolute_percent.append(workingcount / totalcount)

        average_percent_iteration = sum(percent_set_iteration) / len(percent_set_iteration)
        average_abs_percent_iteration = sum(absolute_percent) / len(absolute_percent)

        average_percent.append(average_percent_iteration)
        average_abs.append(average_abs_percent_iteration)

        writer.add_scalar('Validation/Absolute Accuracy', average_abs[k], global_step_loss[k])
        writer.add_scalar('Validation/Relative Error', average_percent[k], global_step_loss[k])
        writer.add_scalar('Validation/Loss: MSE', loss_set_iteration[k], global_step_loss[k])
        epoch_count += 1
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"\nTraining for epoch {k + 1} completed.\n")
        print(f"Time taken for one epoch: {epoch_duration} seconds")

    writer.close()

    Path_model = os.path.join(model_path, f"exp{exp_number}_Model_Path")
    torch.save(model.state_dict(), Path_model)



    print(f"Loss between actual and predicted validation set (Largest Epoch): \n{loss_set_iteration}\n")
    print(f"Average Absolute Percent difference (Largest Epoch): \n{average_percent[-1]}\n")

    x_length_validate_predictions = []
    for i in range(len(predictions_validation[-1])):
        x_length_validate_predictions.append(i + 1)
    print(f"The maximum value of percent difference in last iteration: \n{max(percent_set_iteration[-len(x_length_validate_predictions):])}")
    print(f"The maximum value of percent error averaged on one epoch is respect to actual validation house price is: \n{max(average_percent)}")
    print(f"All average loss values for validation set over each epoch value: {loss_set_iteration}")
    print(f"The minimum value of loss: \n{min(loss_set_iteration)} occurs at epoch number: {loss_set_iteration.index(min(loss_set_iteration)) + 1}")

    x_iterate_plot = []
    x_total_batch_plot = []

    for i in range(len(average_percent)):
        x_iterate_plot.append(i + 1)
    for i in range(len(total_batch_epoch_accuracy)):
        x_total_batch_plot.append(i + 1)

    plt.figure(figsize=(10, 5))
    plt.title('Relative Percent Error Validation vs Training (Y is Relative Percent Error, X is time, each interval we add a new epoch')
    plt.plot(x_iterate_plot, average_percent, label='Validation Relative Percent Error (Y) vs Epoch (X)')
    plt.plot(x_total_batch_plot, total_batch_epoch_accuracy, label='Training Relative Percent Error (Y) vs Epoch (X)')
    plt.legend(title="Legend", loc='upper right', fontsize='x-small')
    plt.show()

    plt.title(f'Absolute Percent Error Validation vs Training with {confidence*100} percent accuracy (Y is Relative Percent Error, X is time, each interval we add a new epoch')
    plt.plot(x_iterate_plot, average_abs, label='Validation Absolute Percent Error (Y) vs Epoch (X)')
    plt.legend(title="Legend", loc='upper right', fontsize='x-small')
    plt.show()

    plt.title('Loss Validation vs Training (Y is Loss, X is time, each interval we add a new epoch')
    plt.plot(x_iterate_plot, loss_set_iteration, label=f'Validation Loss (Y) vs Epoch (X) With Minimum Loss: {min(loss_set_iteration)} and Epoch {loss_set_iteration.index(min(loss_set_iteration)) + 1}')
    plt.plot(x_total_batch_plot, total_batch_epoch_loss, label='Training Loss (Y) vs Epoch (X)')
    plt.legend(title="Legend", loc='upper right', fontsize='x-small')
    plt.show()


class CustomTensorDataset(Dataset):
    def __init__(self, features, targets, device):
        self.device = device
        self.features = torch.tensor(features, dtype=torch.float).to(self.device)
        self.targets = torch.tensor(targets, dtype=torch.float).to(self.device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, placement):
        self.placement = torch.tensor(placement, dtype=torch.float).to(self.device)

        sample = self.features[placement]
        target_final = self.targets[placement]
        return sample, target_final


class CustomTestDataset(Dataset):
    def __init__(self, features, device):
        self.device = device
        self.features = torch.tensor(features, dtype=torch.float).to(self.device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, placement):
        sample = self.features[placement]
        return sample


class MLP_Model(torch.nn.Module):
    def __init__(self, device, attibutes):
        super(MLP_Model, self).__init__()
        self.device = device
        self.attibutes = torch.tensor(attibutes, dtype=torch.float).to(self.device)
        print(attibutes)
        # Define layers correctly without trailing commas
        self.layer1 = nn.Linear(attibutes, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer4 = nn.Linear(1024, 1024)
        self.layer5 = nn.Linear(1024, 1024)
        self.layer6 = nn.Linear(1024, 1)

        self.model_version = nn.PReLU()
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)
        nn.init.xavier_uniform_(self.layer5.weight)
        nn.init.xavier_uniform_(self.layer6.weight)
    def forward(self, x):
        print(x)
        print(x.shape)
        x = self.model_version(self.layer1(x))
        x = self.model_version(self.layer2(x))
        x = self.model_version(self.layer3(x))
        x = self.model_version(self.layer4(x))
        x = self.model_version(self.layer5(x))
        x = self.model_version(self.layer6(x))

        return x.squeeze()

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim, device):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim
        self.device = device

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=False, nonlinearity='relu')

        self.compute_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, self.hidden_dim).to(self.device)
        out, hn = self.rnn(x, h0)
        out = self.compute_linear(out)
        return out.squeeze()


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim, device):
        super(GRUModel, self).__init__()
        self.device = device

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim

        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.layer1 = nn.Linear(hidden_dim, 1024)
        self.layer2 = nn.Linear(1024, output_dim)
        self.model_version = nn.PReLU()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, self.hidden_dim).to(x.device)

        out, _ = self.gru(x, h0)
        out = self.model_version(self.layer1(out))
        out = self.model_version(self.layer2(out))

        return out.squeeze()

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim, device):
        super(LSTM_Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.layer1 = nn.Linear(hidden_dim, 1024)
        self.layer2 = nn.Linear(1024, output_dim)
        self.model_version = nn.PReLU()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.model_version(self.layer1(out))
        out = self.model_version(self.layer2(out))

        return out.squeeze()


def train_one_epoch(model, optimizer, data_loader, global_step, global_step_loss, confidence, loss_function, device, exp_number, std_sales, mean_sales, writer):
    model.train().to(device)
    accuracy_relative = []
    loss_list = []
    absolute_working = []
    absolute_total = []

    for train_features, normalized_target_sales in data_loader:
        optimizer.zero_grad()
        predictions = model(train_features)
        loss = loss_function(predictions, normalized_target_sales)
        writer.add_scalar('Training/Loss', loss, global_step)

        loss.backward()
        optimizer.step()

        scale_prediction = predictions * std_sales + mean_sales
        scale_target_sales = normalized_target_sales * std_sales + mean_sales
        accuracy_percent = 100 * abs((scale_prediction - scale_target_sales) / scale_target_sales)

        scale_target_sales = scale_target_sales.tolist()
        scale_prediction = scale_prediction.tolist()

        accurate_reform = []
        for i in range(len(accuracy_percent)):
            accurate_reform.append(accuracy_percent[i].item())
        mean_accuracy = sum(accurate_reform) / len(accurate_reform)

        loss_list.append(loss.item())
        accuracy_relative.append(mean_accuracy)

        for i in range(len(scale_target_sales)):
            totalcount = 0
            workingcount = 0
            percent_within_lower_bound = (confidence) * scale_target_sales[i]
            percent_within_upper_bound = (2 - confidence) * scale_target_sales[i]

            workingcount += percent_within_lower_bound <= scale_prediction[i] <= percent_within_upper_bound
            totalcount += 1

            absolute_working.append(workingcount)
            absolute_total.append(totalcount)

        writer.add_scalar('Training/Relative Percent Error Over Time', mean_accuracy, global_step)
        global_step += 1

    writer.add_scalar('Training/Loss Over Each Epoch', sum(loss_list) / len(loss_list), global_step_loss)
    writer.add_scalar('Training/Relative Error Over Each Epoch', sum(accuracy_relative) / len(accuracy_relative), global_step_loss)
    writer.add_scalar('Training/Absolute Accuracy Over Each Epoch', sum(absolute_working) / len(absolute_total), global_step_loss)

    return sum(loss_list) / len(loss_list), accuracy_relative, loss_list, global_step


def train_all(model, optimizer, loaded_data, global_step, iteration, list, global_step_loss, confidence, loss_function, device, exp_number, std_sales, mean_sales, writer):
    list_ = list
    model_final_epoch = []
    loss_after, batch_accuracy, batch_loss_list, global_step = train_one_epoch(model, optimizer, loaded_data, global_step, global_step_loss, confidence, loss_function, device, exp_number, std_sales, mean_sales, writer)
    model_final_epoch.append(loss_after)
    print(f"Train Epoch Number: {list_[iteration]}, Loss Average for ALL batches: {loss_after}")
    tensor_model_total_batch_loss = torch.tensor(model_final_epoch).float()
    return tensor_model_total_batch_loss, batch_accuracy, batch_loss_list, global_step

