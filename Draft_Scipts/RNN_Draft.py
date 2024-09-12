import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import argparse
import pathlib
from torch.utils.data import Dataset, DataLoader #here, the pytorch ALREADY HAS pre-defined functions that can be used through the Dataset and DataLoader functions.
from torch.utils.tensorboard import SummaryWriter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

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
test_size=0.315
random_state=42
batch_size = 8



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
df['DateTime'] = pd.to_datetime(df['DateTime'])
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

#print(np.log10(before_numbers_train['Coast_WeatherZone']))

square_df = pd.DataFrame()



for column in before_numbers_train.columns:
    square_df[f"{column}_squared"] = before_numbers_train[column]**2
for column in before_numbers_train.columns:  #this here takes the CUBE OF THE SQUARE of the previous term
    square_df[f"{column}_cube"] = before_numbers_train[column]**3
for column in before_numbers_train.columns:
    square_df[f"{column}_quartic"] = before_numbers_train[column]**4
for column in before_numbers_train.columns:
    square_df[f"{column}_quintic"] = before_numbers_train[column]**5

for column in before_numbers_train.columns:
    square_df[f"{column}_sqrt"] = before_numbers_train[column] ** 0.5
for column in before_numbers_train.columns:
    square_df[f"{column}_log"] = np.log10(before_numbers_train[column])



before_numbers_train_concat_data = pd.concat([before_numbers_train, square_df], axis = 1)    #we CANNOT normalize dummy variables

#
# print(before_numbers_train_concat_data)
#
# csv_file_path = '/media/jacksong/Data1/electricity_price_prediction/Electricity_Prediction_Scripts/Model_Building_Scripts/engineered_powers.csv'
# before_numbers_train_concat_data.to_csv(csv_file_path, index=True)




before_split_train_manipulated = before_numbers_train_concat_data.apply(lambda x: (x - x.mean()) / (x.std()))

# print(before_split_train_manipulated['WGRPP_PANHANDLE_WindPowerProductionGeographical_log'])


concat_data = pd.concat([ohe_weekdays, ohe_months, ohe_hours, before_split_train_manipulated], axis = 1)    #we CANNOT normalize dummy variables


concat_data = concat_data.select_dtypes(exclude=['object'])   #we have to do this since some log values after being normalized go to NaN for some reason
concat_data.fillna(0, inplace=True)



# before_sales_column = before_sales_column.convert_dtypes(exclude=['object'])   #we have to do this since some log values after being normalized go to NaN for some reason
# before_sales_column.fillna(0, inplace=True)


stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size, random_state=random_state)
sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size, random_state=random_state)





# train test split. Size of train data is 80% and size of test data is 20%.

print(f"ttttttttttttttttttttttttttttttttttttttttttttttttt{stat_model}")

input = stat_model.shape[0]
attibutes = stat_model.shape[1]
print(f"The Number of attributes is: {attibutes}")




# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.tensor(stat_model.values).float()
targetsTrain = torch.tensor(sales_column.values).float().type(torch.LongTensor)

# print(f"ffffffffffffffffffffffffffffffff{featuresTrain}")
# mean_sales=targetsTrain.mean()
# std_sales=targetsTrain.std()
# normalized_targetsTrain = (targetsTrain-mean_sales)/std_sales
#
# mean_sales=featuresTrain.mean()
# std_sales=featuresTrain.std()
# normalized_featuresTrain = (featuresTrain-mean_sales)/std_sales

featuresTest = torch.tensor(stat_model_validation.values).float()
targetsTest = torch.tensor(validation_sales.values).float().type(torch.LongTensor)
# create feature and targets tensor for test set.
# featuresTrain = torch.from_numpy(stat_model)
# targetsTrain = torch.from_numpy(sales_column).type(torch.LongTensor) # data type is long
# featuresTest = torch.from_numpy(stat_model_validation)
# targetsTest = torch.from_numpy(validation_sales).type(torch.LongTensor) # data type is long




# batch_size, epoch and iteration
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
# Pytorch train and test sets

train = CustomTensorDataset(featuresTrain,targetsTrain)   #use the normal values, we do not have to normalize
test = CustomTensorDataset(featuresTest,targetsTest)

# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)


# these are data values, not pictures concat_data,                                                                            before_sales_column,


# Create RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN  //this class by definition expects 3 inputs
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first = True, nonlinearity='relu')

        # Readout layer
        self.compute_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, self.hidden_dim).to(device)

        # One time step
        out, hn = self.rnn(x, h0)
        out = self.compute_linear(out)
        print(f"dddddddddddddddddddddddddddddddddddddddd{out}")   #these predictions WILL BE of the sales price

        return out.squeeze()


# # Pytorch train and test sets
# train = TensorDataset(featuresTrain, targetsTrain)
# test = TensorDataset(featuresTest, targetsTest)
#
# # data loader
# train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)



# Create RNN
global_step = 0
input_dim = attibutes  # input dimension
hidden_dim = 100  # hidden layer dimension
layer_dim = 1  # number of hidden layers
output_dim = 1  # output dimension

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

# Cross Entropy Loss
error = nn.MSELoss()

# SGD Optimizer
learning_rate = 0.05
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
seq_dim = 28
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

# input = stat_model.shape[0]
# attibutes = stat_model.shape[1]
# def train_one_epoch(model, optimizer, data_loader,global_step):
sales_column_tensor = torch.tensor(sales_column.values).float()    #this is after split the data
mean_sales=sales_column_tensor.mean()
std_sales=sales_column_tensor.std()


writer = SummaryWriter(f"log_rnn/exp{exp_number}")
global_step_epoch = 1
for epoch in range(num_epochs):
    model.train().to(device)
    accuracy_relative = []
    loss_list = []
    absolute_working = []
    absolute_total = []

    print(f"Training epoch {epoch + 1}: \n -------------------------------------------------------------------")

    for normalized_train_features, target_sales in train_loader:
        optimizer.zero_grad()
        # print(normalized_train_features)
        predictions = model(normalized_train_features).to(device)   #here we want to input a value equal  #or, use predictions

        normalized_target_sales = (target_sales - mean_sales)/std_sales
        #print(f"dddddddddddddddddddddddddddddddddddddddd{predictions}")   #these predictions WILL BE of the sales price
        #print
        #normalized_target_sales = (target_sales - mean_sales)/std_sales
        normalized_predictions = (predictions - mean_sales)/std_sales


        print(f"fffffffffffffffffffffffffffffff{normalized_predictions.squeeze()}")


        #print(predictions)
        #print(normalized_target_sales) #this is a batch size...why is the predicitons NOT?

        loss = loss_function(normalized_predictions, normalized_target_sales)
        print(loss)
        #raise ValueError

        writer.add_scalar('Training/Loss', loss, global_step)

        loss.backward()
        optimizer.step()

        scale_prediction = predictions * std_sales + mean_sales
        accuracy_percent = 100 * abs((scale_prediction - target_sales) / target_sales)

        accurate_reform = []   #we take the average over ALL THE BATCHES that we feed into the rnn
        for i in range(0, len(accuracy_percent)):
            #print(f"ffffffffffffffffffffffff{accuracy_percent.tolist()}")  #we can use detach because we are not using this anymore to update the loss value
            manipulated_list = accuracy_percent.tolist()
            for i in range (0,len(manipulated_list)):
                accurate_reform.append(manipulated_list[i])
        mean_accuracy = sum(accurate_reform) / len(accurate_reform)
        #raise ValueError

        loss_list.append(loss.item())
        accuracy_relative.append(mean_accuracy)
        #print(scale_prediction.tolist())
        for i in range(0, len(target_sales.tolist())):
            totalcount = 0
            workingcount = 0
            percent_within_lower_bound = (confidence) * target_sales.tolist()[i]
            percent_within_upper_bound = (2 - confidence) * target_sales.tolist()[i]

            #print(len(target_sales.tolist()))
            print(f"ddddddddddddddddddddddddddddddddd{scale_prediction.tolist()}")


            workingcount += percent_within_lower_bound <= scale_prediction.tolist()[i] <= percent_within_upper_bound
            totalcount += 1

            absolute_working.append(workingcount)
            absolute_total.append(totalcount)

        writer.add_scalar('Training/Relative Percent Error Over Time', mean_accuracy, global_step)
        global_step += 1
        # raise ValueError

    total_accuracy_relative.append(accuracy_relative)
    total_loss_list.append(total_loss_list)
    total_absolute_working.append(total_absolute_working)
    total_absolute_total.append(total_absolute_total)
    print(f"Loss: {sum(loss_list) / len(loss_list)}")


    writer.add_scalar('Training/Loss Over Each Epoch', sum(loss_list) / len(loss_list), global_step_epoch)
    writer.add_scalar('Training/Relative Error Over Each Epoch', sum(accuracy_relative) / len(accuracy_relative),
                      global_step_epoch)
    writer.add_scalar('Training/Absolute Error Over Each Epoch', sum(absolute_working) / len(absolute_total),
                      global_step_epoch)
    global_step_epoch += 1



    model.eval()
    predictions_validation_iteration = []

    # with torch.no_grad():
    #     for features in test_loaded_data:  #we do this because we load the data in batches, the features is the batches
    #         for i in range(0,len(features)):
    #             outputs = model(features[i])
    #             outputs = outputs.to(device)
    #             final_append=(((outputs * std_sales) + mean_sales).float().item())
    #             predictions_iteration.append(final_append)  # Collecting predictions
    # predictions_.append(predictions_iteration)
    with torch.no_grad():
        for features in test_loader:
            outputs = model(features[0])
            outputs = outputs.to(device)
            # final_value=((outputs * std_validation) + mean_validation)
            for i in range(0, len(outputs)):
                predictions_validation_iteration.append(
                    outputs[i].item())  # we do this because of the way the tensors loaded into the data
            predictions_validation.append(predictions_validation_iteration)
    # --------------------------------------
    tensor_validaion_iteration = torch.tensor(predictions_validation_iteration).float()
    tensor_validation_sales = torch.tensor(validation_sales.tolist()).float()

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
        percent_within_lower_bound = (confidence) * validation_sales.tolist()[i]
        percent_within_upper_bound = (2 - confidence) * validation_sales.tolist()[i]

        workingcount += percent_within_lower_bound <= predictions_validation_iteration[i] <= percent_within_upper_bound
        totalcount += 1
        differance = predictions_validation_iteration[i] - validation_sales.tolist()[i]
        percent_set_iteration.append(abs(100 * (differance / validation_sales.tolist()[i])))
        absolute_percent.append(workingcount / totalcount)

    average_percent_iteration = sum(percent_set_iteration) / len(percent_set_iteration)
    average_abs_percent_iteration = sum(absolute_percent) / len(absolute_percent)

    average_percent.append(average_percent_iteration)
    average_abs.append(average_abs_percent_iteration)
    writer.add_scalar('Validation/Absolute Accuracy', average_abs[epoch], global_step_epoch)
    writer.add_scalar('Validation/Relative Error', average_percent[epoch], global_step_epoch)
    writer.add_scalar('Validation/Loss: MSE', loss_set_iteration[epoch], global_step_epoch)
    global_step_epoch += 1

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