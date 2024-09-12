import os
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
import torch.optim as optim

def Linear_Regression_L1(data_path, test_size, random_state, epochs):
    ohe_path = os.path.join(data_path, "Hot_Encoded_Data_Final_DF.csv")
    concat_data = pd.read_csv(ohe_path)
    actual_desired_lambda = 'SystemLambda_ActualValues'
    before_sales_column = concat_data[actual_desired_lambda]
    concat_data = concat_data.drop(columns=[actual_desired_lambda])
    stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size, random_state=random_state)
    sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size,
                                                      random_state=random_state)
    model = Lasso(alpha=1.0, max_iter=1000000)
    model.fit(stat_model, sales_column)
    mse, r2 = evaluate_model(model, stat_model_validation, validation_sales)
    print(f"Lasso Values, MSE: {mse:.4f}, R2: {r2:.4f}")

def Linear_Regression_L2(data_path, test_size, random_state, epochs):
    ohe_path = os.path.join(data_path, "Hot_Encoded_Data_Final_DF.csv")
    concat_data = pd.read_csv(ohe_path)
    actual_desired_lambda = 'SystemLambda_ActualValues'
    before_sales_column = concat_data[actual_desired_lambda]
    concat_data = concat_data.drop(columns=[actual_desired_lambda])
    stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size, random_state=random_state)
    sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size,
                                                      random_state=random_state)
    model = Ridge(alpha=1.0, max_iter=1000000)
    model.fit(stat_model, sales_column)
    mse, r2 = evaluate_model(model, stat_model_validation, validation_sales)
    print(f"Ridge, MSE: {mse:.4f}, R2: {r2:.4f}")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def gather_inputs(data_path):
    ohe_path = os.path.join(data_path, "Hot_Encoded_Data_Final_DF.csv")
    concat_data = pd.read_csv(ohe_path)
    columns_to_drop = [col for col in concat_data.columns if 'ActualValues' in col]
    concat_data = concat_data.drop(columns=columns_to_drop)
    concat_data = concat_data.iloc[:, 1:]
    attibutes = concat_data.shape[1]
    return attibutes

def shift_sequence(stat_model_validation, seq_length):
    validation_features_list = []
    for i in range(len(stat_model_validation) - seq_length):
        validation_features_list.append(stat_model_validation[i:i + seq_length])
    validation_features_list = np.array(validation_features_list)
    validation_features_tensor = torch.tensor(validation_features_list, dtype=torch.float32)
    return validation_features_tensor


def run_model(data_path, model_path, exp_number, number_epochs, test_size, random_state, batch_size, accuracy_range,
              class_model, loss_function, model_params, device, seq_length):
    ohe_path = os.path.join(data_path, "Hot_Encoded_Data_Final_DF.csv")
    concat_data = pd.read_csv(ohe_path)
    no_ohe_path = os.path.join(data_path, "Combined_Data_Before_OHE/Concated_DF_DST_Filtered.csv")
    concat_data_noohe = pd.read_csv(no_ohe_path)
    actual_desired_lambda = 'SystemLambda_ActualValues'
    before_sales_column = concat_data[actual_desired_lambda]
    before_sales_column_unnoralized = concat_data_noohe[actual_desired_lambda]
    stat_model_std_mean, stat_model_validation_std_mean = train_test_split(before_sales_column_unnoralized, test_size=test_size, random_state=random_state)
    columns_to_drop = [col for col in concat_data.columns if 'ActualValues' in col]
    concat_data = concat_data.drop(columns=columns_to_drop)
    concat_data = concat_data.iloc[:, 1:]

    if not model_params:
        stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size, random_state=random_state) #, shuffle = False)#
        sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size, random_state=random_state)
    else:
        stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size,shuffle = False)#
        sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size, shuffle = False)
    attibutes = stat_model.shape[1]
    train_features = torch.tensor(stat_model.values).float()
    validation_features = torch.tensor(stat_model_validation.values).float()
    sales_column_tensor = torch.tensor(sales_column.values).float()
    validation_sales_tensor = torch.tensor(validation_sales.values).float()
    sales_column_tensor_unnormalized = torch.tensor(stat_model_std_mean.values).float() #this did not get mixed up
    sales_column_validation_tensor_unnormalized = torch.tensor(stat_model_validation_std_mean.values).float() #this did not get mixed up

    mean_sales = sales_column_tensor_unnormalized.mean()
    std_sales = sales_column_tensor_unnormalized.std()
    if not model_params:
        train_dataset_custom_dataset = CustomTensorDataset_MLP_Linear(train_features, sales_column_tensor)
        validation_dataset_custom_dataset = CustomTensorDataset_MLP_Linear(validation_features, validation_sales_tensor)
        train_loaded_data = DataLoader(train_dataset_custom_dataset, batch_size=batch_size, shuffle=True)
        validation_loaded_data = DataLoader(validation_dataset_custom_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_features_normalized = shift_sequence(stat_model, seq_length)
        validation_features_normalized = shift_sequence(stat_model_validation, seq_length)
        validation_sales_tensor_normalized = shift_sequence(validation_sales_tensor, seq_length)
        sales_column_tensor_normalized = shift_sequence(sales_column_tensor, seq_length)
        train_dataset = torch.utils.data.TensorDataset(train_features_normalized, sales_column_tensor_normalized)
        train_loaded_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(validation_features_normalized, validation_sales_tensor_normalized)
        validation_loaded_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if not model_params:  # case for MLP
        print(f"Using {class_model} model")
        model = class_model(attibutes).to(device)
    else:
        print(f"Using {class_model} model")
        model = class_model(**model_params).to(device)
    writer = SummaryWriter(f"log/exp{exp_number}")
    global_step_batch = 1
    global_step_epoch = 1
    epoch_count = 0
    inital_weight_decay = 1e-7
    initial_learning_rate = 1e-7
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=inital_weight_decay)
    for k in range(number_epochs):
        print(f"-------------------------------------\nTraining the {k + 1} epoch:\n")
        start_time = time.time()
        loss_after, global_step_batch = train_one_epoch(model, optimizer, train_loaded_data, global_step_batch, global_step_epoch,
                                                  accuracy_range, loss_function, device, std_sales, mean_sales,
                                                  writer, model_params)
        print(f"Train Epoch Number: {k+1}, Loss average all batches of epoch: {loss_after}")
        predictions_validation_batch = []
        normalized_validation_batch = []
        model.eval()
        with torch.no_grad():
            if not model_params:
                for features, _ in validation_loaded_data:
                    features = features.to(device)
                    outputs = model(features)  #we do not use [0] since we need to UNPACK everything from the dataloader, or else we get a list of unpacked tensors
                    final_value = ((outputs * std_sales) + mean_sales)
                    for i in range(len(final_value)):
                        predictions_validation_batch.append(final_value[i].item())
                        normalized_validation_batch.append(outputs[i].item())
            else:
                for features, normalized_outputs in validation_loaded_data:
                    features = features.to(device)
                    outputs = model(features)
                    final_value = ((outputs * std_sales) + mean_sales)
                    for i in range(len(final_value)):
                        predictions_validation_batch.append(final_value[i][-1][-1].item())
                        normalized_validation_batch.append(outputs[i][-1][-1].item())
        tensor_validaion_iteration = torch.tensor(normalized_validation_batch).float()
        predictions_validation_batch_tensor = torch.tensor(predictions_validation_batch).float()
        if not model_params:
            loss_validation = loss_function(tensor_validaion_iteration, validation_sales_tensor)
            dummy_variable = validation_sales_tensor
        else:
            validation_sales_tensor_normalized = validation_sales_tensor_normalized.unsqueeze(-1)
            print(validation_sales_tensor_normalized.shape)
            print(tensor_validaion_iteration.shape)
            loss_validation = loss_function(tensor_validaion_iteration, validation_sales_tensor_normalized)
            dummy_variable = validation_sales_tensor_normalized
        percent_within_lower_bound = (accuracy_range) * dummy_variable
        percent_within_upper_bound = (2 - accuracy_range) * dummy_variable
        within_bounds = (predictions_validation_batch_tensor >= percent_within_lower_bound) & (predictions_validation_batch_tensor <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(dummy_variable)
        differance_batch = abs(dummy_variable - predictions_validation_batch_tensor)
        abs_differance_validate = torch.mean(differance_batch)
        relative_error_batch_tensor = (abs(100 * (differance_batch / dummy_variable)))
        relative_error_batch = torch.mean(relative_error_batch_tensor)
        writer.add_scalar('Validation/Absolute Accuracy', (workingcount)/(totalcount), global_step_epoch)
        writer.add_scalar('Validation/Relative Error', (relative_error_batch), global_step_epoch)
        writer.add_scalar('Validation/Loss: MSE', loss_validation.item(), global_step_epoch)
        writer.add_scalar('Validation/Absolute Differance In Price', (abs_differance_validate), global_step_epoch)
        epoch_count += 1
        global_step_epoch += 1
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"\nTraining for epoch {k + 1} completed.\n")
        print(f"Time taken for one epoch: {epoch_duration} seconds")
    writer.close()
    Path_model = os.path.join(model_path, f"exp{exp_number}_Model_Path")
    torch.save(model.state_dict(), Path_model)


class CustomTensorDataset_MLP_Linear(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float)
        self.targets = torch.tensor(targets, dtype=torch.float)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        sample = self.features[index]
        target_final = self.targets[index]
        return sample, target_final


class CustomTensorDataset(Dataset):
    def __init__(self, features, targets, seq_length):  #more data proccessing
        self.seq_length = seq_length  #more data proccessing
        self.features = torch.tensor(features, dtype=torch.float)
        self.targets = torch.tensor(targets, dtype=torch.float)

    def __len__(self):
        return len(self.features)- self.seq_length

    def __getitem__(self, index):
        feature_seq = self.features[index:index+self.seq_length]
        target_seq = self.targets[index:index+self.seq_length]
        return feature_seq, target_seq


class CustomTestDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        sample = self.features[index]
        return sample


class MLP_Model(torch.nn.Module):
    def __init__(self, attibutes):
        super(MLP_Model, self).__init__()
        self.attibutes = attibutes
        self.layer1 = nn.Linear(attibutes, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer4 = nn.Linear(1024, 1024)
        self.layer5 = nn.Linear(1024, 1024)
        self.layer6 = nn.Linear(1024, 1)
        self.activation_function = nn.PReLU()
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
        x = self.activation_function(self.layer6(x))
        return x.squeeze()


class Linear_Regessor_Pytorch(torch.nn.Module):
    def __init__(self, attibutes):
        super(Linear_Regessor_Pytorch, self).__init__()
        self.attibutes = attibutes
        self.layer1 = nn.Linear(attibutes, 1024)
        self.layer2 = nn.Linear(1024, 1)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x):
        x = (self.layer1(x))
        x = (self.layer2(x))
        return x.squeeze()


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.compute_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.rnn(x, h0)
        out = self.compute_linear(out)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.layer1 = nn.Linear(hidden_dim, 1024)
        self.layer2 = nn.Linear(1024, output_dim)
        self.activation_function = nn.PReLU()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.activation_function(self.layer1(out))
        out = self.activation_function(self.layer2(out))
        return out


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.layer1 = nn.Linear(hidden_dim, 1024)
        self.layer2 = nn.Linear(1024, output_dim)
        self.activation_function = nn.PReLU()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.activation_function(self.layer1(out))
        out = self.activation_function(self.layer2(out))
        return out

def train_one_epoch(model, optimizer, data_loader, global_step_batch, global_step_epoch, accuracy_range, loss_function, device, std_sales, mean_sales, writer, model_params):
    model.train()
    error_relative_list = []
    loss_all_batch = []
    accurate_working = 0
    accurate_total = 0
    differance_all_batches_list = []
    for train_features, normalized_target_sales in data_loader:
        optimizer.zero_grad()
        train_features = train_features.to(device)
        normalized_target_sales = normalized_target_sales.to(device)
        predictions = model(train_features)
        if model_params:
            normalized_target_sales = normalized_target_sales.unsqueeze(-1)
        loss = loss_function(predictions, normalized_target_sales)
        writer.add_scalar('Training/Loss each batch', loss, global_step_batch)
        loss.backward()
        optimizer.step()
        scale_prediction = predictions * std_sales + mean_sales
        scale_target_sales = normalized_target_sales * std_sales + mean_sales
        relative_error_percent = 100 * abs((scale_prediction - scale_target_sales) / scale_target_sales)
        absolute_differance_batch = abs((scale_prediction - scale_target_sales))
        mean_differance = torch.mean(absolute_differance_batch)
        differance_all_batches_list.append(mean_differance)
        error_relative = torch.mean(relative_error_percent)
        error_relative_list.append(error_relative)
        writer.add_scalar('Training/Absolute Differance In Price (batch)', mean_differance, global_step_batch)
        loss_all_batch.append(loss.item())
        percent_within_lower_bound = (accuracy_range) * scale_target_sales
        percent_within_upper_bound = (2 - accuracy_range) * scale_target_sales
        within_bounds = (scale_prediction >= percent_within_lower_bound) & (scale_prediction <= percent_within_upper_bound)
        workingcount = np.sum(within_bounds.int().tolist())
        totalcount = len(scale_prediction)
        accurate_working += workingcount
        accurate_total += totalcount
        writer.add_scalar('Training/Relative Percent Error Over Time (Each Batch)', (error_relative), global_step_batch)
        global_step_batch += 1
    writer.add_scalar('Training/Relative Error Over Each Epoch (Avergae All Batches)', sum(error_relative_list)/len(error_relative_list), global_step_epoch)
    writer.add_scalar('Training/Absolute Accuracy Over Each Epoch (Average Over All Batches)', (accurate_working) / (accurate_total), global_step_epoch)
    writer.add_scalar('Training/Absolute Differance per Epoch (Last Batch)', sum(differance_all_batches_list)/len(differance_all_batches_list), global_step_epoch)
    return sum(loss_all_batch)/len(loss_all_batch), global_step_batch