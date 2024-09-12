import os
import time
import inspect

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge


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


class CustomDataset_MLP_Linear(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        sample = self.features[index]
        target_final = self.targets[index]
        return sample, target_final


class CustomTensorDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.seq_length = seq_length
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
    def __init__(self, attributes):
        super(MLP_Model, self).__init__()
        self.attributes = attributes
        self.layer1 = nn.Linear(attributes, 1024)
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
    def __init__(self, attributes):
        super(Linear_Regessor_Pytorch, self).__init__()
        self.attributes = attributes
        self.layer1 = nn.Linear(attributes, 1024)
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
        self.layer1 = nn.Linear(hidden_dim, 1024)
        self.layer2 = nn.Linear(1024, output_dim)
        self.activation_function = nn.PReLU()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.rnn(x, h0)
        out = self.activation_function(self.layer1(out))
        out = self.activation_function(self.layer2(out))
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

def Linear_Regression_L1(data_path, test_size, random_state, validate_size):
    ohe_path = os.path.join(data_path, "Hot_Encoded_Data_Final_DF.csv")
    concat_data_normalized = pd.read_csv(ohe_path)
    sys_lambda_column_name = 'SystemLambda_ActualValues'
    normalized_sys_lambda = concat_data_normalized[sys_lambda_column_name]
    concat_data_normalized = concat_data_normalized.drop(columns=[normalized_sys_lambda])
    columns_to_drop = [col for col in concat_data_normalized.columns if 'ActualValues' in col]
    concat_data_normalized = concat_data_normalized.drop(columns=columns_to_drop)
    concat_data_normalized = concat_data_normalized.iloc[:, 1:]
    no_ohe_path = os.path.join(data_path, "Combined_Data_Before_OHE/Concated_DF_DST_Filtered.csv")
    concat_data_noohe_unnormalized = pd.read_csv(no_ohe_path)
    un_normalized_system_lambda = concat_data_noohe_unnormalized[sys_lambda_column_name]
    columns_to_drop_unnormalized = [col for col in concat_data_noohe_unnormalized.columns if 'ActualValues' in col]
    concat_data_noohe_unnormalized = concat_data_noohe_unnormalized.drop(columns=columns_to_drop_unnormalized)
    concat_data_noohe_unnormalized = concat_data_noohe_unnormalized.iloc[:, 1:]
    random_split_boolean = False
    x_train, x_validate, x_test = train_validate_test_split(concat_data_normalized, test_size, random_state, validate_size, random_split_boolean)
    y_train, y_validate, y_test = train_validate_test_split(normalized_sys_lambda, test_size, random_state, validate_size, random_split_boolean)
    _, _, x_test_unnormalized = train_validate_test_split(concat_data_noohe_unnormalized, test_size, random_state, validate_size, random_split_boolean)
    _, _, y_test_unnormalized = train_validate_test_split(un_normalized_system_lambda, test_size, random_state, validate_size, random_split_boolean)
    model = Lasso(alpha=1.0, max_iter=1000000)
    model.fit(x_train, y_train)
    mse, r2 = evaluate_model(model, x_validate, y_validate)
    print(f"Lasso Values, MSE: {mse:.4f}, R2: {r2:.4f}")
    return x_test, y_test, x_test_unnormalized, y_test_unnormalized

def Linear_Regression_L2(data_path, test_size, random_state, validate_size):
    ohe_path = os.path.join(data_path, "Hot_Encoded_Data_Final_DF.csv")
    concat_data_normalized = pd.read_csv(ohe_path)
    sys_lambda_column_name = 'SystemLambda_ActualValues'
    normalized_sys_lambda = concat_data_normalized[sys_lambda_column_name]
    concat_data_normalized = concat_data_normalized.drop(columns=[normalized_sys_lambda])
    columns_to_drop = [col for col in concat_data_normalized.columns if 'ActualValues' in col]
    concat_data_normalized = concat_data_normalized.drop(columns=columns_to_drop)
    concat_data_normalized = concat_data_normalized.iloc[:, 1:]
    no_ohe_path = os.path.join(data_path, "Combined_Data_Before_OHE/Concated_DF_DST_Filtered.csv")
    concat_data_noohe_unnormalized = pd.read_csv(no_ohe_path)
    un_normalized_system_lambda = concat_data_noohe_unnormalized[sys_lambda_column_name]
    columns_to_drop_unnormalized = [col for col in concat_data_noohe_unnormalized.columns if 'ActualValues' in col]
    concat_data_noohe_unnormalized = concat_data_noohe_unnormalized.drop(columns=columns_to_drop_unnormalized)
    concat_data_noohe_unnormalized = concat_data_noohe_unnormalized.iloc[:, 1:]
    random_split_boolean = False
    x_train, x_validate, x_test = train_validate_test_split(concat_data_normalized, test_size, random_state, validate_size, random_split_boolean)
    y_train, y_validate, y_test = train_validate_test_split(normalized_sys_lambda, test_size, random_state, validate_size, random_split_boolean)
    _, _, x_test_unnormalized = train_validate_test_split(concat_data_noohe_unnormalized, test_size, random_state, validate_size, random_split_boolean)
    _, _, y_test_unnormalized = train_validate_test_split(un_normalized_system_lambda, test_size, random_state, validate_size, random_split_boolean)
    model = Ridge(alpha=1.0, max_iter=1000000)
    model.fit(x_train, y_train)
    mse, r2 = evaluate_model(model, x_validate, y_validate)
    print(f"Ridge, MSE: {mse:.4f}, R2: {r2:.4f}")
    return x_test, y_test, x_test_unnormalized, y_test_unnormalized

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def gather_inputs(data_path):
    ohe_path = os.path.join(data_path, "Hot_Encoded_Data_Final_DF.csv")
    concat_data_normalized = pd.read_csv(ohe_path)
    columns_to_drop = [col for col in concat_data_normalized.columns if 'ActualValues' in col]
    concat_data_normalized = concat_data_normalized.drop(columns=columns_to_drop)
    concat_data_normalized = concat_data_normalized.iloc[:, 1:]
    attributes = concat_data_normalized.shape[1]
    return attributes

def shift_sequence(tensor, seq_length):
    append_element_list = []
    for i in range(len(tensor) - seq_length):
        append_element_list.append(tensor[i:i + seq_length])
    append_element_list = np.array(append_element_list)
    sequence_tensor = torch.tensor(append_element_list, dtype=torch.float32)
    return sequence_tensor

def train_validate_test_split(data, test_size, random_state, validate_size, random_split_boolean):
    temp, test = train_test_split(data, test_size=test_size, shuffle= random_split_boolean, random_state=random_state)
    train, validate = train_test_split(temp, test_size=validate_size, shuffle= random_split_boolean, random_state=random_state)
    return train, validate, test

def manipulate_data(data_path, validate_size, test_size, random_state, batch_size, class_model, model_params, device, seq_length):
    caller_frame = inspect.stack()[1]
    caller_file = os.path.basename(caller_frame.filename)
    ohe_path = os.path.join(data_path, "Hot_Encoded_Data_Final_DF.csv")
    concat_data_normalized = pd.read_csv(ohe_path)
    no_ohe_path = os.path.join(data_path, "Combined_Data_Before_OHE/Concated_DF_DST_Filtered.csv")
    concat_data_noohe_unnormalized = pd.read_csv(no_ohe_path)
    sys_lambda_column_name = 'SystemLambda_ActualValues'
    normalized_system_lambda = concat_data_normalized[sys_lambda_column_name]
    un_normalized_system_lambda = concat_data_noohe_unnormalized[sys_lambda_column_name]
    columns_to_drop = [col for col in concat_data_normalized.columns if 'ActualValues' in col]
    concat_data_normalized = concat_data_normalized.drop(columns=columns_to_drop)
    concat_data_normalized = concat_data_normalized.iloc[:, 1:]
    columns_to_drop = [col for col in concat_data_noohe_unnormalized.columns if 'ActualValues' in col]
    concat_data_noohe_unnormalized = concat_data_noohe_unnormalized.drop(columns=columns_to_drop)
    concat_data_noohe_unnormalized = concat_data_noohe_unnormalized.iloc[:, 1:]
    if not model_params:
        random_split_boolean = False
        x_train, x_validate, x_test = train_validate_test_split(concat_data_normalized, test_size, random_state, validate_size, random_split_boolean)
        y_train, y_validate, y_test = train_validate_test_split(normalized_system_lambda, test_size, random_state, validate_size, random_split_boolean)
        _, _, x_test_unnormalized = train_validate_test_split(concat_data_noohe_unnormalized, test_size, random_state, validate_size, random_split_boolean)
        y_train_unnormalized, y_validate_unnormalized, y_test_unnormalized = train_validate_test_split(un_normalized_system_lambda, test_size, random_state, validate_size, random_split_boolean)
    else:
        random_split_boolean = False
        x_train, x_validate, x_test = train_validate_test_split(concat_data_normalized, test_size, random_state, validate_size, random_split_boolean)
        y_train, y_validate, y_test = train_validate_test_split(normalized_system_lambda, test_size, random_state, validate_size, random_split_boolean)
        _, _, x_test_unnormalized = train_validate_test_split(concat_data_noohe_unnormalized, test_size, random_state, validate_size, random_split_boolean)
        y_train_unnormalized, y_validate_unnormalized, y_test_unnormalized = train_validate_test_split(un_normalized_system_lambda, test_size, random_state, validate_size, random_split_boolean)
    attributes = x_train.shape[1]
    x_train_tensor = torch.tensor(x_train.values).float()
    x_validate_tensor = torch.tensor(x_validate.values).float()
    y_train_tensor = torch.tensor(y_train.values).float()
    y_validate_tensor = torch.tensor(y_validate.values).float()
    train_un_normalized_sys_lam_tensor = torch.tensor(y_train_unnormalized.values).float()
    validate_un_normalized_sys_lam_tensor = torch.tensor(y_validate_unnormalized.values).float()
    mean_sys_lam = train_un_normalized_sys_lam_tensor.mean()
    std_sys_lam = train_un_normalized_sys_lam_tensor.std()
    if not model_params:
        train_dataset_custom_dataset = CustomTensorDataset_MLP_Linear(x_train_tensor, y_train_tensor)
        validation_dataset_custom_dataset = CustomTensorDataset_MLP_Linear(x_validate_tensor, y_validate_tensor)
        train_loaded_data = DataLoader(train_dataset_custom_dataset, batch_size=batch_size, shuffle=True)
        validation_loaded_data = DataLoader(validation_dataset_custom_dataset, batch_size=batch_size, shuffle=False)
    else:
        x_train_tensor_normalized_seq = shift_sequence(x_train, seq_length)
        validation_features_normalized_seq = shift_sequence(x_validate, seq_length)
        validation_sys_lam_tensor_normalized_seq = shift_sequence(y_validate_tensor, seq_length)
        validate_un_normalized_sys_lam_seq = shift_sequence(y_validate_unnormalized, seq_length)
        y_train_tensor_seq = shift_sequence(y_train_tensor, seq_length)
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor_normalized_seq, y_train_tensor_seq)
        train_loaded_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(validation_features_normalized_seq, validation_sys_lam_tensor_normalized_seq)
        validation_loaded_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if 'lightning_model_class_functions.py' or 'copy_lightning.py' in caller_file:
        return x_train, y_train, x_validate, y_validate, x_test, y_test, std_sys_lam, mean_sys_lam

    else:
        if not model_params:
            print(f"Using {class_model} model")
            model = class_model(attributes).to(device)
        else:
            print(f"Using {class_model} model")
            model = class_model(**model_params).to(device)
    if not model_params:
        return model, validation_loaded_data, train_loaded_data, std_sys_lam, mean_sys_lam, validate_un_normalized_sys_lam_tensor, y_validate_tensor, x_test, y_test, x_test_unnormalized, y_test_unnormalized
    else:
        return model, validation_loaded_data, train_loaded_data, validate_un_normalized_sys_lam_seq, std_sys_lam, mean_sys_lam, validate_un_normalized_sys_lam_tensor, validation_sys_lam_tensor_normalized_seq, x_test, y_test, x_test_unnormalized, y_test_unnormalized

def extract_loaded_data(model, validation_loaded_data, device, std_sys_lam, mean_sys_lam, model_params):
    predictions_validation_unnormalized = []
    normalized_validation_predictions = []
    model.eval()
    with torch.no_grad():
        for features, _ in validation_loaded_data:
            outputs = model(features.to(device))
            outputs = outputs.to(device)
            unnormalized_final_value = ((outputs * std_sys_lam) + mean_sys_lam)
            if not model_params:
                for i in range(len(unnormalized_final_value)):
                    predictions_validation_unnormalized.append(unnormalized_final_value[i].item())
                    normalized_validation_predictions.append(outputs[i].item())
            else:
                for i in range(len(unnormalized_final_value)):
                    predictions_validation_unnormalized.append(unnormalized_final_value[i][-1].item())
                    normalized_validation_predictions.append(outputs[i][-1].item())
    normalized_validation_predictions_tensor = torch.tensor(normalized_validation_predictions).float()
    predictions_validation_unnormalized_tensor = torch.tensor(predictions_validation_unnormalized).float()
    return normalized_validation_predictions_tensor, predictions_validation_unnormalized_tensor

def validate_accuracy_calculations(accuracy_range, dummy_variable, predictions_validation_unnormalized_tensor):
    percent_within_lower_bound = (accuracy_range) * dummy_variable
    percent_within_upper_bound = (2 - accuracy_range) * dummy_variable
    within_bounds = (predictions_validation_unnormalized_tensor >= percent_within_lower_bound) & ( predictions_validation_unnormalized_tensor <= percent_within_upper_bound)
    workingcount = np.sum(within_bounds.int().tolist())
    totalcount = len(dummy_variable)
    abs_differance_validation_all = abs(dummy_variable - predictions_validation_unnormalized_tensor)
    mean_abs_differance_validate = torch.mean(abs_differance_validation_all)
    relative_error_tensor = (abs(100 * (abs_differance_validation_all / dummy_variable)))
    relative_error_tensor_mean = torch.mean(relative_error_tensor)
    return relative_error_tensor_mean, mean_abs_differance_validate, totalcount, workingcount

def train_accuracy_calculations(accuracy_range, actual_sys_lam, un_normalized_prediction_sys_lam):
    percent_within_lower_bound = (accuracy_range) * actual_sys_lam
    percent_within_upper_bound = (2 - accuracy_range) * actual_sys_lam
    within_bounds = (un_normalized_prediction_sys_lam >= percent_within_lower_bound) & (un_normalized_prediction_sys_lam <= percent_within_upper_bound)
    workingcount = np.sum(within_bounds.int().tolist())
    totalcount = len(within_bounds.int().tolist())
    relative_error_percent = 100 * abs((un_normalized_prediction_sys_lam - actual_sys_lam) / actual_sys_lam)
    absolute_differance_batch = abs((un_normalized_prediction_sys_lam - actual_sys_lam))
    mean_differance_batch = torch.mean(absolute_differance_batch)
    mean_error_relative = torch.mean(relative_error_percent)
    return mean_differance_batch, mean_error_relative, workingcount, totalcount

def run_model(data_path, model_path, exp_number, number_epochs, validate_size, test_size, random_state, batch_size, accuracy_range, class_model, loss_function, model_params, device, seq_length):
    if not model_params:
        model, validation_loaded_data, train_loaded_data, std_sys_lam, mean_sys_lam, validate_un_normalized_sys_lam_tensor, y_validate_tensor, x_test, y_test, x_test_unnormalized, y_test_unnormalized = manipulate_data(
            data_path, validate_size, test_size, random_state, batch_size, class_model, model_params, device, seq_length)
    else:
        model, validation_loaded_data, train_loaded_data, validate_un_normalized_sys_lam_seq, std_sys_lam, mean_sys_lam, validate_un_normalized_sys_lam_tensor, validation_sys_lam_tensor_normalized_seq, x_test, y_test, x_test_unnormalized, y_test_unnormalized = manipulate_data(
            data_path, validate_size, test_size, random_state, batch_size, class_model, model_params, device, seq_length)
    writer = SummaryWriter(f"log/exp{exp_number}")
    global_step_batch = 1
    global_step_epoch = 1
    epoch_count = 0
    inital_weight_decay = 0
    initial_learning_rate = 1e-5
    decay_rate = 0.96
    decay_steps = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=inital_weight_decay)
    for k in range(number_epochs):
        if epoch_count <= 15:
            initial_learning_rate = initial_learning_rate
        else:
            initial_learning_rate = initial_learning_rate * (decay_rate ** (epoch_count / decay_steps))
        print(f"-------------------------------------\nTraining the {k + 1} epoch:\n")
        start_time = time.time()
        avg_epoch_loss, global_step_batch = train_one_epoch(model, optimizer, train_loaded_data, global_step_batch, global_step_epoch, accuracy_range, loss_function, device, std_sys_lam, mean_sys_lam, writer, model_params)
        print(f"Train Epoch Number: {k+1}, Loss average all batches of epoch: {avg_epoch_loss}")
        normalized_validation_predictions_tensor, predictions_validation_unnormalized_tensor = extract_loaded_data(model, validation_loaded_data, device, std_sys_lam, mean_sys_lam, model_params)
        if not model_params:
            loss_validation = loss_function(normalized_validation_predictions_tensor, y_validate_tensor)
            dummy_variable = validate_un_normalized_sys_lam_tensor
        else:
            loss_validation = loss_function(normalized_validation_predictions_tensor, validation_sys_lam_tensor_normalized_seq[:,-1])
            dummy_variable = validate_un_normalized_sys_lam_seq[:,-1]
        relative_error_tensor_mean, mean_abs_differance_validate, totalcount, workingcount = validate_accuracy_calculations(accuracy_range, dummy_variable, predictions_validation_unnormalized_tensor)
        writer.add_scalar('Validation/Absolute Accuracy', (workingcount)/(totalcount), global_step_epoch)
        writer.add_scalar('Validation/Relative Error', (relative_error_tensor_mean), global_step_epoch)
        writer.add_scalar('Validation/Loss: MSE', loss_validation.item(), global_step_epoch)
        writer.add_scalar('Validation/Absolute Differance In Price', (mean_abs_differance_validate), global_step_epoch)
        epoch_count += 1
        global_step_epoch += 1
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"\nTraining for epoch {k + 1} completed.\n")
        print(f"Time taken for one epoch: {epoch_duration} seconds")
        writer.add_scalar('Time/Time taken for one epoch, training and validation analysis', (epoch_duration), global_step_epoch)
    writer.close()
    Path_model = os.path.join(model_path, f"exp{exp_number}_Model_Path")
    torch.save(model.state_dict(), Path_model)
    return x_test, y_test, x_test_unnormalized, y_test_unnormalized, x_train_tensor, x_validate_tensor, y_train_tensor, y_validate_tensor

def train_one_epoch(model, optimizer, data_loader, global_step_batch, global_step_epoch, accuracy_range, loss_function, device, std_sys_lam, mean_sys_lam, writer, model_params):
    model.train()
    error_relative_list = []
    train_loss_batch_list = []
    accurate_working = 0
    accurate_total = 0
    differance_all_batches_list = []
    for train_features, normalized_train_sys_lam in data_loader:
        optimizer.zero_grad()
        train_features = train_features.to(device)
        normalized_train_sys_lam = normalized_train_sys_lam.to(device)
        predictions = model(train_features)
        if model_params:
            normalized_train_sys_lam = normalized_train_sys_lam.unsqueeze(-1)
        train_loss = loss_function(predictions, normalized_train_sys_lam)
        writer.add_scalar('Training/Loss each batch', train_loss, global_step_batch)
        train_loss.backward()
        optimizer.step()
        if not model_params:
            un_normalized_prediction_sys_lam = predictions * std_sys_lam + mean_sys_lam
            actual_sys_lam = normalized_train_sys_lam * std_sys_lam + mean_sys_lam
        else:
            un_normalized_prediction_sys_lam = predictions[:,-1,-1] * std_sys_lam + mean_sys_lam
            actual_sys_lam = normalized_train_sys_lam[:,-1,-1] * std_sys_lam + mean_sys_lam
        train_loss_batch_list.append(train_loss.item())
        mean_differance_batch, mean_error_relative, workingcount, totalcount = train_accuracy_calculations(accuracy_range, actual_sys_lam, un_normalized_prediction_sys_lam)
        accurate_working += workingcount
        accurate_total += totalcount
        differance_all_batches_list.append(mean_differance_batch)
        error_relative_list.append(mean_error_relative)
        writer.add_scalar('Training/Absolute Differance In Price (batch)', mean_differance_batch, global_step_batch)
        writer.add_scalar('Training/Relative Percent Error Over Time (Each Batch)', (mean_error_relative), global_step_batch)
        global_step_batch += 1
    writer.add_scalar('Training/Relative Error Over Each Epoch (Avergae All Batches)', sum(error_relative_list)/len(error_relative_list), global_step_epoch)
    writer.add_scalar('Training/Absolute Accuracy Over Each Epoch (Average Over All Batches)', (accurate_working) / (accurate_total), global_step_epoch)
    writer.add_scalar('Training/Absolute Differance per Epoch (Last Batch)', sum(differance_all_batches_list)/len(differance_all_batches_list), global_step_epoch)
    return sum(train_loss_batch_list)/len(train_loss_batch_list), global_step_batch