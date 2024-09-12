import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# def sklearn_linear(data_path, test_size, random_state, epochs, exp_number, batch_size):
#     ohe_path = os.path.join(data_path, "Hot_Encoded_Data_Final_DF.csv")
#     concat_data = pd.read_csv(ohe_path)
#     actual_desired_lambda = 'SystemLambda_ActualValues'
#     before_sales_column = concat_data[actual_desired_lambda]
#     concat_data = concat_data.drop(columns=[actual_desired_lambda])
#     stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size, random_state=random_state)
#     sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size, random_state=random_state)
#     global_step = 0
#     mse_list = []
#     r2_list = []
#     x_step = []
#     writer = SummaryWriter(f"log/exp{exp_number}")
#     model = Linear_Regression(stat_model, sales_column, stat_model_validation, validation_sales)
#     num_batches = len(stat_model) // batch_size + (1 if len(stat_model) % batch_size != 0 else 0)
#     for epoch in range(epochs):
#         print(f"--------------------------------------------------------\nTraining for epoch {epoch + 1} started.\n")
#         for i in range(num_batches):
#             start_idx = i * batch_size
#             end_idx = (i + 1) * batch_size
#             X_batch = stat_model[start_idx:end_idx]
#             y_batch = sales_column[start_idx:end_idx]
#             model = Linear_Regression(X_batch, y_batch, stat_model_validation, validation_sales)
#             model.train_model()
#         print(f"\nTraining for epoch {epoch + 1} completed.\n")
#         mse, r2 = model.evaluate_model()
#         writer.add_scalar('Validation/Absolute Accuracy', mse, global_step)
#         writer.add_scalar('Validation/Relative Error', r2, global_step)
#         mse_list.append(mse)
#         r2_list.append(r2)
#         global_step += 1
#     for i in range(len(mse_list)):
#         x_step.append(i + 1)
#     plt.figure(figsize=(10, 5))
#     plt.title('Loss and Accuracies During Training')
#     plt.plot(x_step, r2_list, label='R2 During Training')
#     plt.plot(x_step, mse_list, label='MSE Loss Training')
#     plt.legend(title="Legend", loc='upper right', fontsize='x-small')
#     plt.show()


class L1Regularization(nn.Module):
    def __init__(self, base_model, lambda_l1):
        super(L1Regularization, self).__init__()
        self.base_model = base_model
        self.lambda_l1 = lambda_l1

    def forward(self, x):
        return self.base_model(x)

    def l1_loss(self):
        l1_loss = 0
        for param in self.base_model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.lambda_l1 * l1_loss

def gather_inputs(data_path):
    ohe_path = os.path.join(data_path, "Hot_Encoded_Data_Final_DF.csv")
    concat_data = pd.read_csv(ohe_path)
    columns_to_drop = [col for col in concat_data.columns if 'ActualValues' in col]
    concat_data = concat_data.drop(columns=columns_to_drop)
    concat_data = concat_data.iloc[:, 1:]
    attibutes = concat_data.shape[1]
    return attibutes

def run_model(data_path, model_path, exp_number, number_epochs, test_size, random_state, batch_size, confidence,
              class_model, loss_function, model_params, device, seq_length, L1_boolean):
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
    stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size, random_state=random_state)#,shuffle
    sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size, random_state=random_state)
    attibutes = stat_model.shape[1]
    inputs = stat_model.shape[0]
    train_features = torch.tensor(stat_model.values).float()
    validation_features = torch.tensor(stat_model_validation.values).float()
    sales_column_tensor = torch.tensor(sales_column.values).float()
    validation_sales_tensor = torch.tensor(validation_sales.values).float()
    sales_column_tensor_unnormalized = torch.tensor(stat_model_std_mean.values).float() #this did not get mixed up
    mean_sales = sales_column_tensor_unnormalized.mean()
    std_sales = sales_column_tensor_unnormalized.std()
    if not model_params:
        print("Using MLP or Linear Regression")
        train_dataset_custom_dataset = CustomTensorDataset_MLP_Linear(train_features, sales_column_tensor, device)
        validation_dataset_custom_dataset = CustomTensorDataset_MLP_Linear(validation_features, validation_sales_tensor, device)
    else:
        train_dataset_custom_dataset = CustomTensorDataset(train_features, sales_column_tensor, device, seq_length)
        validation_dataset_custom_dataset = CustomTensorDataset(validation_features, validation_sales_tensor, device,
                                                                seq_length)
    train_loaded_data = DataLoader(train_dataset_custom_dataset, batch_size=batch_size, shuffle=True)
    validation_loaded_data = DataLoader(validation_dataset_custom_dataset, batch_size=batch_size, shuffle=False)
    if not model_params:  # case for MLP
        model = class_model(device, attibutes).to(device)
    else:
        model = class_model(**model_params).to(device)



    writer = SummaryWriter(f"log/exp{exp_number}")
    global_step_batch = 1
    global_step_epoch = 1

    epoch_count = 0
    inital_weight_decay = 1e-7
    decay_rate = 0.96
    decay_steps = 10
    initial_learning_rate = 1e-5

    for k in range(number_epochs):
        if epoch_count <= 15:
            learning_rate = initial_learning_rate
        else:
            learning_rate = initial_learning_rate * (decay_rate ** (epoch_count / decay_steps))

        if epoch_count <= 15:
            weight_decay = inital_weight_decay
        else:
            weight_decay = inital_weight_decay * (decay_rate ** (epoch_count / decay_steps))
        relative_error_batch = []
        abs_differance_validate = []
        print(f"-------------------------------------\nTraining the {k + 1} epoch:\n")
        start_time = time.time()
        if (class_model==Linear_Regessor_Pytorch and L1_boolean == "True"): #here, we must implement the L1 regularization manually during testing
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0)
        elif class_model==Linear_Regessor_Pytorch:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_after, global_step_batch = train_one_epoch(model, optimizer, train_loaded_data, global_step_batch, global_step_epoch,
                                                  confidence, loss_function, device, std_sales, mean_sales,
                                                  writer, L1_boolean)
        print(f"Train Epoch Number: {k+1}, Loss average all batches of epoch: {loss_after}")
        predictions_validation_iteration = []
        loss_validation_list = []
        model.eval()
        with torch.no_grad():
            for features in validation_loaded_data:
                outputs = model(features[0])
                outputs = outputs.to(device)
                final_value = ((outputs * std_sales) + mean_sales)
                for i in range(len(final_value)):
                    predictions_validation_iteration.append(final_value[i].item())
                    loss_validation_list.append(outputs[i].item())

        tensor_validaion_iteration = torch.tensor(loss_validation_list).float()
        loss_validation = loss_function(tensor_validaion_iteration, validation_sales_tensor)
        totalcount = 0
        workingcount = 0
        for i in range(len(predictions_validation_iteration)):
            percent_within_lower_bound = (confidence) * sales_column_tensor_unnormalized[i]
            percent_within_upper_bound = (2 - confidence) * sales_column_tensor_unnormalized[i]
            workingcount += percent_within_lower_bound <= predictions_validation_iteration[i] <= percent_within_upper_bound  # we use the MEAN value of the training data in the split
            totalcount += 1
            differance_batch = abs(sales_column_tensor_unnormalized[i] - predictions_validation_iteration[i])
            abs_differance_validate.append(abs(sales_column_tensor_unnormalized[i] - predictions_validation_iteration[i]))
            relative_error_batch.append(abs(100 * (differance_batch / sales_column_tensor_unnormalized[i])))   #THIS IS WRONG
        writer.add_scalar('Validation/Absolute Accuracy', (workingcount)/(totalcount), global_step_epoch)
        writer.add_scalar('Validation/Relative Error', sum(relative_error_batch)/len(relative_error_batch), global_step_epoch)
        writer.add_scalar('Validation/Loss: MSE', loss_validation.item(), global_step_epoch)
        writer.add_scalar('Validation/Absolute Differance In Price', sum(abs_differance_validate)/len(abs_differance_validate), global_step_epoch)
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
    def __init__(self, features, targets, device):
        self.device = device
        self.features = torch.tensor(features, dtype=torch.float).to(self.device)
        self.targets = torch.tensor(targets, dtype=torch.float).to(self.device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        self.idx = torch.tensor(idx, dtype=torch.float).to(self.device)

        sample = self.features[idx]
        target_final = self.targets[idx]
        return sample, target_final
class CustomTensorDataset(Dataset):
    def __init__(self, features, targets, device, seq_length):
        self.device = device
        self.seq_length = seq_length
        self.features = torch.tensor(features, dtype=torch.float).to(self.device)
        self.targets = torch.tensor(targets, dtype=torch.float).to(self.device)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        self.idx = torch.tensor(idx, dtype=torch.float).to(self.device)
        feature = self.features[idx]
        target_final = self.targets[idx]
        feature = feature.unsqueeze(0).repeat(self.seq_length, 1)
        return feature, target_final

class CustomTestDataset(Dataset):
    def __init__(self, features, device):
        self.device = device
        self.features = torch.tensor(features, dtype=torch.float).to(self.device)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        sample = self.features[idx]
        return sample

class Linear_Regression:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = LinearRegression()
        # self.device = device
        self.X_train = X_train  # .to(device)
        self.y_train = y_train  # .to(device)
        self.X_test = X_test  # .to(device)
        self.y_test = y_test  # .to(device)
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2
    def get_coefficients(self):
        return self.model.coef_, self.model.intercept_


class MLP_Model(torch.nn.Module):
    def __init__(self, device, attibutes):
        super(MLP_Model, self).__init__()
        self.device = device
        self.attibutes = torch.tensor(attibutes, dtype=torch.float).to(self.device)
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
        x = self.model_version(self.layer1(x))
        x = self.model_version(self.layer2(x))
        x = self.model_version(self.layer3(x))
        x = self.model_version(self.layer4(x))
        x = self.model_version(self.layer5(x))
        x = self.model_version(self.layer6(x))
        return x.squeeze()

class Linear_Regessor_Pytorch(torch.nn.Module):
    def __init__(self, device, attibutes):
        super(Linear_Regessor_Pytorch, self).__init__()
        self.device = device
        self.attibutes = torch.tensor(attibutes, dtype=torch.float).to(self.device)
        self.layer1 = nn.Linear(attibutes, 1024)
        self.layer2 = nn.Linear(1024, 1)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
    def forward(self, x):
        x = (self.layer1(x))
        x = (self.layer2(x))
        return x.squeeze()

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim, device):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim
        self.device = device
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.compute_linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        out, hn = self.rnn(x, h0)
        out = self.compute_linear(out[:, -1, :])
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
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.gru(x, h0)

        out = self.model_version(self.layer1(out[:, -1, :]))
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
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.model_version(self.layer1(out[:, -1, :]))
        out = self.model_version(self.layer2(out))
        return out.squeeze()
def train_one_epoch(model, optimizer, data_loader, global_step_batch, global_step_epoch, confidence, loss_function, device, std_sales, mean_sales, writer, L1_boolean):
    model.train().to(device)
    error_relative = []
    loss_all_batch = []
    accurate_absolute_working = 0
    accurate_absolute_total = 0
    differance_all_batches = []
    for train_features, normalized_target_sales in data_loader: #data_loader
        optimizer.zero_grad()
        predictions = model(train_features)
        loss = loss_function(predictions, normalized_target_sales)

        if isinstance(L1_boolean == "True"):
            l1_loss = model.l1_loss()
            loss += l1_loss
        writer.add_scalar('Training/Loss each batch', loss, global_step_batch)



        loss.backward()
        optimizer.step()

        scale_prediction = predictions * std_sales + mean_sales
        scale_target_sales = normalized_target_sales * std_sales + mean_sales
        relative_error_percent = 100 * abs((scale_prediction - scale_target_sales) / scale_target_sales)
        scale_target_sales = scale_target_sales.tolist()
        scale_prediction = scale_prediction.tolist()
        relative_error_list = []
        absolute_differance_batch = []
        for i in range(len(relative_error_percent)):
            relative_error_list.append(relative_error_percent[i].item())
            absolute_differance_batch.append(abs(scale_prediction[i] - scale_target_sales[i]))
            differance_all_batches.append(abs(scale_prediction[i] - scale_target_sales[i]))
        mean_differance = sum(absolute_differance_batch)/ len(absolute_differance_batch)
        writer.add_scalar('Training/Absolute Differance In Price (batch)', mean_differance, global_step_batch)
        loss_all_batch.append(loss.item())
        error_relative.append(sum(relative_error_list)/ len(relative_error_list))
        for i in range(len(scale_prediction)):
            totalcount = 0
            workingcount = 0
            percent_within_lower_bound = (confidence) * scale_target_sales[i]
            percent_within_upper_bound = (2 - confidence) * scale_target_sales[i]
            workingcount += percent_within_lower_bound <= scale_prediction[i] <= percent_within_upper_bound
            totalcount += 1
            accurate_absolute_working += (workingcount)
            accurate_absolute_total += (totalcount)
        writer.add_scalar('Training/Relative Percent Error Over Time', sum(relative_error_list)/ len(relative_error_list), global_step_batch)
        global_step_batch += 1
    writer.add_scalar('Training/Relative Error Over Each Epoch (Last Batch)', sum(error_relative)/len(error_relative), global_step_epoch)
    writer.add_scalar('Training/Absolute Accuracy Over Each Epoch (Average Over All Batches)', (accurate_absolute_working) / (accurate_absolute_total), global_step_epoch)
    writer.add_scalar('Training/Absolute Differance per Epoch (Last Batch)', sum(differance_all_batches)/len(differance_all_batches), global_step_epoch)
    return sum(loss_all_batch)/len(loss_all_batch), global_step_batch