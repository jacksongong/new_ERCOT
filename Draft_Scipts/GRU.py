import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pathlib
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings("ignore")


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
    type = str,
    required=True,
    help = "name of csv file with all the data"
)

parser.add_argument(
    "--model_path",
    type = str,
    required=True,
    help = "path to saved model after training"
)

parser.add_argument(
    "--exp_number",
    type = int,
    required=True,
    help = "experiment number under the log folder"
)

parser.add_argument(
    "--number_epochs",
    type = int,
    required=True,
    help = "number of trained epochs"
)


args = parser.parse_args()
data_path = args.data_path
name_csv = args.name_csv
model_path = args.model_path
exp_number = args.exp_number
number_epochs = args.number_epochs


df_function = os.path.join(data_path, name_csv)
test_size=0.155  #we predict on one year of future data (24 hours*366 days)
random_state=42



pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device} device\n----------------------------------------------------------------------------------------")

df= pd.read_csv(df_function)

common_time = 'Hour_Ending'
common_delivery = 'DeliveryDate'
actual_desired_lambda = 'SystemLambda_ActualValues'


before_sales_column = df[actual_desired_lambda]
df['DateTime'] = pd.to_datetime(df['DeliveryDate'])  #this is used for the hour models

#df['DateTime'] = pd.to_datetime(df['DateTime'])
df['weekday'] = df['DateTime'].dt.weekday
ohe_weekdays = pd.get_dummies(df['weekday'], prefix='weekday')
ohe_weekdays = ohe_weekdays.astype(bool).astype(int)
df['month'] = df['DateTime'].dt.month
ohe_months = pd.get_dummies(df['month'], prefix='month')
ohe_months = ohe_months.astype(bool).astype(int)
df['hour'] = df['DateTime'].dt.hour
ohe_hours = pd.get_dummies(df['hour'], prefix='hour')
ohe_hours = ohe_hours.astype(bool).astype(int)
train_drop = df.drop(columns=['weekday', 'month', 'hour', common_delivery, common_time, actual_desired_lambda, 'DateTime'])



train_drop = train_drop.iloc[:, 1:]
before_numbers_train = train_drop.select_dtypes(exclude=['object'])
before_numbers_train.fillna(0, inplace=True)
square_df = pd.DataFrame()



for column in before_numbers_train.columns:
    square_df[f"{column}_squared"] = before_numbers_train[column]**2
for column in before_numbers_train.columns:
    square_df[f"{column}_cube"] = before_numbers_train[column]**3
for column in before_numbers_train.columns:
    square_df[f"{column}_quartic"] = before_numbers_train[column]**4
for column in before_numbers_train.columns:
    square_df[f"{column}_quintic"] = before_numbers_train[column]**5

for column in before_numbers_train.columns:
    square_df[f"{column}_sqrt"] = before_numbers_train[column] ** 0.5
for column in before_numbers_train.columns:
    square_df[f"{column}_log"] = np.log10(before_numbers_train[column])


before_numbers_train_concat_data = pd.concat([before_numbers_train, square_df], axis = 1)

before_split_train_manipulated = before_numbers_train_concat_data.apply(lambda x: (x - x.mean()) / (x.std()))
concat_data = pd.concat([ohe_weekdays, ohe_months, ohe_hours, before_split_train_manipulated], axis = 1)


concat_data = concat_data.select_dtypes(exclude=['object'])
concat_data.fillna(0, inplace=True)


stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size, random_state=random_state)    #for the RNN, we could experiment with NOT shuffling the
sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size, random_state=random_state)
print(validation_sales)

#print(f"ttttttttttttttttttttttttttttttttttttttttttttttttt{stat_model}")

input = stat_model.shape[0]
attibutes = stat_model.shape[1]
print(f"The Number of attributes is: {attibutes}")


featuresTrain = torch.tensor(stat_model.values).float()
targetsTrain = torch.tensor(sales_column.values).float().type(torch.LongTensor)

featuresTest = torch.tensor(stat_model_validation.values).float()
targetsTest = torch.tensor(validation_sales.values).float().type(torch.LongTensor)

validation_sales = validation_sales.tolist()

batch_size = 100
num_epochs = number_epochs

class CustomTensorDataset(Dataset):
    def __init__(self,features,targets):
        self.features = torch.tensor(features, dtype=torch.float).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float).to(device)
    def __len__(self):
        return len(self.features)

    def __getitem__(self, placement):
        sample = self.features[placement]
        target_final = self.targets[placement]
        return sample, target_final

class CustomTestDataset(Dataset):
    def __init__(self,features):
        self.features = torch.tensor(features, dtype=torch.float).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, placement):
        sample = self.features[placement]
        return sample

train = CustomTensorDataset(featuresTrain,targetsTrain)
test = CustomTensorDataset(featuresTest,targetsTest)

train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

# class RNNModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#         super(RNNModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.layer_dim = layer_dim
#         self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first = False, nonlinearity='relu')
#         self.compute_linear = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         h0 = torch.zeros(self.layer_dim, self.hidden_dim).to(device)
#         out, hn = self.rnn(x, h0)
#         out = self.compute_linear(out)
#         #print(f"dddddddddddddddddddddddddddddddddddddddd{out}")
#
#         return out.squeeze()


# class RNNModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim,
#                  seq_dim, output_dim):
#         super(RNNModel, self).__init__()
#
#         self.hidden_dim = hidden_dim
#         self.layer_dim = layer_dim
#
#         self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True)
#         self.fc1 = nn.Linear(hidden_dim * seq_dim, output_dim)
#
#     def forward(self, x):
#         h0 = torch.zeros(self.layer_dim, self.hidden_dim).to(device=device)
#         c0 = torch.zeros(self.layer_dim, self.hidden_dim).to(device=device)
#         print(output_dim)
#         print((hidden_dim * seq_dim))
#         out, _ = self.lstm(x, (h0, c0))
#         print(out.shape)
#         out = self.fc1(out)
#         print(out)
#         return out.squeeze()


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim, output_dim):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim

        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first = True)

        # Example: Adding multiple fully connected layers
        # self.fc1 = nn.Linear(hidden_dim * seq_dim, 128)
        # self.fc2 = nn.Linear(128, output_dim)

        self.layer1 = nn.Linear(hidden_dim, 1024)
        # self.dropout1 = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(1024, output_dim)
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.layer3 = nn.Linear(1024, 1024)
        # # self.dropout3 = nn.Dropout(p=0.2)
        # self.layer4 = nn.Linear(1024, 1024)
        # # self.dropout4 = nn.Dropout(p=0.2)
        # self.layer5 = nn.Linear(1024, 1024)
        #
        # self.layer6 = nn.Linear(1024, 1)

        self.model_version = nn.PReLU()
        # nn.init.xavier_uniform_(self.layer1.weight)
        # nn.init.xavier_uniform_(self.layer2.weight)
        # nn.init.xavier_uniform_(self.layer3.weight)
        # nn.init.xavier_uniform_(self.layer4.weight)
        # nn.init.xavier_uniform_(self.layer5.weight)
        # nn.init.xavier_uniform_(self.layer6.weight)

    def forward(self, x):
        # batch_size = x.size(0)

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.layer_dim,  self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.gru(x, h0)

        # Reshape output from (batch_size, seq_dim, hidden_dim) to (batch_size, seq_dim*hidden_dim)
        #print(out)
        # Pass through fully connected layers

        # x = self.layer1(x)
        # x = self.dropout1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        out = self.model_version(self.layer1(out))
        # # #x = self.dropout1(x)
        out = self.model_version(self.layer2(out))
        # #x = self.dropout2(x)
        # x = self.model_version(self.layer3(x))
        # # #x = self.dropout3(x)
        # x = self.model_version(self.layer4(x))
        # # x = self.dropout4(x)
        # x = self.model_version(self.layer5(x))
        #
        # x = self.model_version(self.layer6(x))


        # out = self.fc1(out)
        # out = torch.relu(out)
        # out = self.fc2(out)

        return out.squeeze()

global_step = 0
input_dim = attibutes
hidden_dim = 100
layer_dim = 5  #square of this?
output_dim = 1
seq_dim = 28

model = RNNModel(input_dim, hidden_dim, layer_dim, seq_dim, output_dim).to(device)
error = nn.MSELoss()

learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_list = []
iteration_list = []
accuracy_list = []
count = 0
confidence = 0.80
loss_function=nn.MSELoss()


total_accuracy_relative = []
total_loss_list = []
total_absolute_working = []
total_absolute_total = []


predictions_validation = []
loss_set_iteration = []
percent_set_iteration = []
absolute_percent = []
average_percent=[]
average_abs=[]
total_batch_epoch_accuracy=[]
total_batch_epoch_loss=[]

sales_column_tensor = torch.tensor(sales_column.values).float()
mean_sales=sales_column_tensor.mean()
std_sales=sales_column_tensor.std()

writer = SummaryWriter(f"log/exp{exp_number}")
global_step_epoch = 1

for epoch in range(num_epochs):
    model.train().to(device)
    loss_list = []  # why is this loss staying the same and not changing---this will contain the values of the batch

    accuracy_relative = []
    absolute_working = []
    absolute_total = []
    print(f"Training epoch {epoch + 1}: \n")
    start_time_lag = time.time()

    for normalized_train_features, target_sales in train_loader:
        optimizer.zero_grad()
        predictions = model(normalized_train_features).to(device)
        normalized_target_sales = (target_sales - mean_sales)/std_sales
        normalized_predictions = (predictions - mean_sales)/std_sales



        #print(f"fffffffffffffffffffffffffffffff{normalized_predictions.squeeze()}")

        loss = loss_function(normalized_predictions, normalized_target_sales)

        #print(loss)

        writer.add_scalar('Training/Loss', loss, global_step)
        loss.backward()
        optimizer.step()
        scale_prediction = predictions * std_sales + mean_sales

        accuracy_percent = 100 * abs((scale_prediction - target_sales) / target_sales)

        scale_prediction = scale_prediction.tolist()
        target_sales = target_sales.tolist()

        accuracy_percent = accuracy_percent.tolist()
        accurate_reform = []
        for i in range(0, len(accuracy_percent)):
            for i in range (0,len(accuracy_percent)):
                accurate_reform.append(accuracy_percent[i])
        mean_accuracy = sum(accurate_reform) / len(accurate_reform)

        loss_list.append(loss.item())

        #print(loss_list)

        accuracy_relative.append(mean_accuracy)
        for i in range(0, len(target_sales)):
            totalcount = 0
            workingcount = 0
            percent_within_lower_bound = (confidence) * target_sales[i]
            percent_within_upper_bound = (2 - confidence) * target_sales[i]
            #print(f"ddddddddddddddddddddddddddddddddd{scale_prediction}")


            workingcount += percent_within_lower_bound <= scale_prediction[i] <= percent_within_upper_bound
            totalcount += 1

            absolute_working.append(workingcount)
            absolute_total.append(totalcount)

        writer.add_scalar('Training/Relative Percent Error Over Time', mean_accuracy, global_step)
        global_step += 1


    total_accuracy_relative.append(accuracy_relative)
    total_loss_list.append(total_loss_list)
    total_absolute_working.append(total_absolute_working)
    total_absolute_total.append(total_absolute_total)
    print(f"Loss: {sum(loss_list) / len(loss_list)}")




    writer.add_scalar('Training/Loss Over Each Epoch', sum(loss_list) / len(loss_list), global_step_epoch)

    non_inf_values = [x for x in accuracy_relative if not (x == float('inf') or x == float('-inf'))]
    average = sum(non_inf_values) / len(non_inf_values)
    accuracy_relative = [x if not (x == float('inf') or x == float('-inf')) else average for x in accuracy_relative]
    #print(accuracy_relative)


    writer.add_scalar('Training/Relative Error Over Each Epoch', sum(accuracy_relative) / len(accuracy_relative),
                      global_step_epoch)
    writer.add_scalar('Training/Absolute Error Over Each Epoch', sum(absolute_working) / sum(absolute_total),  #this is the same thing as sum(absolute_total)
                      global_step_epoch)
    #global_step_epoch += 1

    model.eval()
    predictions_validation_iteration = []

    with torch.no_grad():
        for features in test_loader:
            outputs = model(features[0])
            outputs = outputs.to(device)
            for i in range(0, len(outputs)):
                predictions_validation_iteration.append(
                    outputs[i].item())  # we do this because of the way the tensors loaded into the data
            predictions_validation.append(predictions_validation_iteration)
    # --------------------------------------
    tensor_validaion_iteration = torch.tensor(predictions_validation_iteration).float()
    tensor_validation_sales = torch.tensor(validation_sales).float()

    mean_tensor_validaion_iteration = tensor_validaion_iteration.mean()
    std_tensor_validaion_iteration = tensor_validaion_iteration.std()
    mean_tensor_validation_sales = tensor_validation_sales.mean()
    std_tensor_validation_sales = tensor_validation_sales.std()

    normalized_tensor_validaion_iteration = ( tensor_validaion_iteration - mean_tensor_validaion_iteration) / std_tensor_validaion_iteration
    normalized_tensor_validation_sales = (tensor_validation_sales - mean_tensor_validation_sales) / std_tensor_validation_sales

    loss_validation = loss_function(normalized_tensor_validaion_iteration, normalized_tensor_validation_sales)
    loss_set_iteration.append(loss_validation.item())

    for i in range(0, len(predictions_validation_iteration)):
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
    writer.add_scalar('Validation/Absolute Accuracy', average_abs[epoch], global_step_epoch)
    writer.add_scalar('Validation/Relative Error', average_percent[epoch], global_step_epoch)
    writer.add_scalar('Validation/Loss: MSE', loss_set_iteration[epoch], global_step_epoch)
    global_step_epoch += 1

    end_time_lag = time.time()
    epoch_duration_lag = end_time_lag - start_time_lag
    print(f"Time between epochs: {epoch_duration_lag} seconds\n-------------------------------------------")

writer.close()


Path_model = os.path.join(model_path, f"exp{exp_number}_Model_Path")
torch.save(model.state_dict(), Path_model)



print(f"Loss between actual and predicted validation set (Largest Epoch): \n{loss_set_iteration}\n")
print(f"Average Absolute Percent differance (Largest Epoch): \n{average_percent[-1]}")
#------------------------

x_length_validate_predictions = []
for i in range(0, len(predictions_validation[-1])):
    x_length_validate_predictions.append(i + 1)
print(
    f"The maximum value of percent differance in last iteration: \n{max(percent_set_iteration[-len(x_length_validate_predictions):])}")  # if we do not use the [] at the end, it would be the predictions for EVERYTHING
print(
    f"The maximum value of percent error averaged on one epoch is repect to actual validation house price is: \n{max(average_percent)}")
print(f"All average loss values for validation set over each epoch value: {loss_set_iteration}")
print(
    f"The minimum value of loss: \n{min(loss_set_iteration)} occurs at epoch number: {loss_set_iteration.index(min(loss_set_iteration)) + 1}")  # this is the first value of the loss_set_iteration

x_iterate_plot=[]
x_total_batch_plot=[]

for i in range(0,len(average_percent)):
    x_iterate_plot.append(i+1)
for i in range(0,len(total_batch_epoch_accuracy)):
    x_total_batch_plot.append(i+1)

plt.figure(figsize=(10, 5))
plt.title('Relative Percent Error Validation vs Training (Y is Relative Percent Error, X is time, each interval we add a new epoch')
plt.plot(x_iterate_plot,average_percent, label='Validation Relative Percent Error (Y) vs Epoch (X)')
plt.plot(x_total_batch_plot,total_batch_epoch_accuracy,label='Training Relative Percent Error (Y) vs Epoch (X)')
plt.legend(title="Legend", loc='upper right', fontsize='x-small')
plt.show()

plt.title(f'Absolute Percent Error Validation vs Training with {confidence*100} percent accuracy (Y is Relative Percent Error, X is time, each interval we add a new epoch')
plt.plot(x_iterate_plot, average_abs, label='Validation Absolute Percent Error (Y) vs Epoch (X)')
#plt.plot(x_total_batch_plot, total_batch_epoch_accuracy, label='Training Relative Percent Error (Y) vs Epoch (X)')
plt.legend(title="Legend", loc='upper right', fontsize='x-small')
plt.show()


plt.title('Loss Validation vs Training (Y is Loss, X is time, each interval we add a new epoch')
plt.plot(x_iterate_plot,loss_set_iteration, label=f'Validation Loss (Y) vs Epoch (X) With Minimum Loss: {min(loss_set_iteration)} and Epoch {loss_set_iteration.index(min(loss_set_iteration))+1}')
plt.plot(x_total_batch_plot,total_batch_epoch_loss, label='Training Loss (Y) vs Epoch (X)')
plt.legend(title="Legend", loc='upper right', fontsize='x-small')
plt.show()