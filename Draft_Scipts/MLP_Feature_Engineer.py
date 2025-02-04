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
test_size=0.155
random_state=42
batch_size = 64    #a smaller batch size will allow epoch to train faster (almost by degress of 6 when moving from 256 to 8, but make the time between epochs the same (longer 2 sec)



pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
#print(f"Using {device} device\n----------------------------------------------------------------------------------------")

df= pd.read_csv(df_function)

#df_test = pd.read_csv("/media/jacksong/Data1/House_Project_Kaggle-PyTorch/test.csv")
# test_drop = df_test.drop(columns=['Id'])
# only_numbers_test = test_drop.select_dtypes(exclude=['object'])
# only_numbers_test.fillna(0, inplace=True)
# stat_model_test = only_numbers_test.apply(lambda x: (x - x.mean()) / (x.std()))  # we don't take the abs value so we can see what values are less or more than.




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


# csv_file_path = '/media/jacksong/Data1/electricity_price_prediction/Electricity_Prediction_Scripts/Model_Building_Scripts/logarithm_final.csv'
# concat_data.to_csv(csv_file_path, index=True)


stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size, random_state=random_state)  #we train on ANY, we predict on the validation in order
sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size, shuffle=False, random_state=random_state)


input = stat_model.shape[0]
attibutes = stat_model.shape[1]
#print(f"The Number of attributes is: {attibutes}")


class CustomTensorDataset(Dataset):
    def __init__(self,features,targets):
        self.features = torch.tensor(features, dtype=torch.float).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float).to(device)
    def __len__(self):
        return len(self.features)

    def __getitem__(self, placement):
        self.placement = torch.tensor(placement, dtype=torch.float).to(device)

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


train_features = torch.tensor(stat_model.values).float()
#test_features = torch.tensor(stat_model_test.values).float()
validation_features = torch.tensor(stat_model_validation.values).float()

sales_column_tensor = torch.tensor(sales_column.values).float()


validation_sales_tensor=torch.tensor(validation_sales.values).float()
mean_validation=validation_sales_tensor.mean()
std_validation=validation_sales_tensor.std()
validation_normalized_sales = (validation_sales_tensor-mean_validation)/std_validation



validation_sales = validation_sales.tolist()




mean_sales=sales_column_tensor.mean()
std_sales=sales_column_tensor.std()
normalized_sales = (sales_column_tensor-mean_sales)/std_sales

normalized_tensor_sales = torch.tensor(normalized_sales).float()

train_dataset_custom_dataset = CustomTensorDataset(train_features, normalized_tensor_sales)
#test_dataset_custom_dataset = CustomTestDataset(test_features)
validation_dataset_custom_dataset = CustomTensorDataset(validation_features, validation_normalized_sales)

train_loaded_data = DataLoader(train_dataset_custom_dataset, batch_size=batch_size, shuffle=True)#, num_workers = 2)#, persistent_workers=True, )
validation_loaded_data = DataLoader(validation_dataset_custom_dataset, batch_size=batch_size, shuffle=False)#, num_workers = 2)#, persistent_workers=True)
#test_loaded_data = DataLoader(test_dataset_custom_dataset, batch_size=batch_size, shuffle=False)

class Train_Model(torch.nn.Module):
    def __init__(self):
        super(Train_Model, self).__init__()

        self.attibutes = torch.tensor(attibutes, dtype=torch.float).to(device)


        # Define layers correctly without trailing commas
        self.layer1 = nn.Linear(attibutes, 1024)
        #self.dropout1 = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(1024, 1024)
        #self.dropout2 = nn.Dropout(p=0.2)
        self.layer3 = nn.Linear(1024, 1024)
        #self.dropout3 = nn.Dropout(p=0.2)
        self.layer4 = nn.Linear(1024, 1024)
        #self.dropout4 = nn.Dropout(p=0.2)
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
        #x = self.layer1(x)
        # x = self.dropout1(x)
        #x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.model_version(self.layer1(x))
        # # #x = self.dropout1(x)
        x = self.model_version(self.layer2(x))
        # #x = self.dropout2(x)
        x = self.model_version(self.layer3(x))
        # #x = self.dropout3(x)
        x = self.model_version(self.layer4(x))
        #x = self.dropout4(x)
        x = self.model_version(self.layer5(x))

        x = self.model_version(self.layer6(x))


        return x.squeeze()
# ----------------------------------------------------------------------------------------

loss_function=nn.MSELoss()
def train_one_epoch(model, optimizer, data_loader,global_step, global_step_loss):
    model.train().to(device)
    accuracy_relative=[]
    loss_list = []
    absolute_working = []
    absolute_total = []
    confidence = 0.80
    for train_features, normalized_target_sales in data_loader:
        optimizer.zero_grad()
        predictions = model(train_features)
        loss = loss_function(predictions, normalized_target_sales)
        writer.add_scalar('Training/Loss', loss, global_step)

        loss.backward()
        optimizer.step()


        scale_prediction=predictions*std_sales+mean_sales
        scale_target_sales=normalized_target_sales*std_sales+mean_sales
        accuracy_percent=100*abs((scale_prediction-scale_target_sales)/scale_target_sales)

        scale_target_sales = scale_target_sales.tolist()
        scale_prediction = scale_prediction.tolist()

        accurate_reform=[]
        for i in range(0, len(accuracy_percent)):
            accurate_reform.append(accuracy_percent[i].item())
        mean_accuracy=sum(accurate_reform)/len(accurate_reform)

        loss_list.append(loss.item())
        accuracy_relative.append(mean_accuracy)



        for i in range(0, len(scale_target_sales)):
            totalcount = 0
            workingcount = 0
            percent_within_lower_bound = (confidence) * scale_target_sales[i]
            percent_within_upper_bound = (2 - confidence) * scale_target_sales[i]

            workingcount += percent_within_lower_bound <= scale_prediction[i] <= percent_within_upper_bound
            totalcount += 1

            absolute_working.append(workingcount)
            absolute_total.append(totalcount)


        writer.add_scalar('Training/Relative Percent Error Over Time', mean_accuracy, global_step)
        global_step+=1

    writer.add_scalar('Training/Loss Over Each Epoch', sum(loss_list) / len(loss_list), global_step_loss)
    writer.add_scalar('Training/Relative Error Over Each Epoch', sum(accuracy_relative) / len(accuracy_relative), global_step_loss)
    writer.add_scalar('Training/Absolute Error Over Each Epoch', sum(absolute_working)/len(absolute_total), global_step_loss)

    return(sum(loss_list)/len(loss_list)), accuracy_relative, loss_list, global_step

def train_all(model, optimizer, loaded_data, global_step, iteration, list, global_step_loss):
    list_=list
    model_final_epoch = []
    loss_after, batch_accuracy, batch_loss_list, global_step = train_one_epoch(model, optimizer, loaded_data,global_step, global_step_loss)
    model_final_epoch.append(loss_after)
    print(f"Train Epoch Number: {list_[iteration]}, Loss Average for ALL batches: {loss_after}")
    tensor_model_total_batch_loss = torch.tensor(model_final_epoch).float()
    return(tensor_model_total_batch_loss),batch_accuracy, batch_loss_list, global_step



if __name__ == '__main__':

    #os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    torch.cuda.empty_cache()
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ['TORCH_USE_CUDA_DSA'] = "1"
    # torch.multiprocessing.set_start_method('spawn', force=True)


    writer = SummaryWriter(f"log/exp{exp_number}")
    global_step_list = [1]

    epochs = number_epochs
    global_step_loss = [1]
    for i in range (0, epochs):
        global_step_loss.append(i+1)
    epoch_count=0
    shown_data = 20
    epoch_list = [i for i in range(1,epochs+1)]
    test_id=5
    isolated_attribut = 'LotArea'
    predictions_ = []
    confidence=0.80
    predictions_validation = []
    loss_set_iteration = []
    percent_set_iteration=[]
    average_percent=[]
    average_abs=[]
    absolute_percent = []
    weight_decay = 1e-2
    total_batch_epoch_accuracy =[]
    total_batch_epoch_loss = []
    # learning_rate = 1e-5

    if epoch_count<=5:
        learning_rate = 1e-5
    else:
        learning_rate=pow(1,-epoch_count)

    model = Train_Model().to(device)





    for k in range (0, epochs):
        print(f"Training the {k+1} epoch:\n")
        start_time = time.time()

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_batch_loss,batch_epoch_accuracy, batch_epoch_loss, global_step = train_all(model, optimizer, train_loaded_data, global_step_list[k],k,epoch_list, global_step_loss[k])
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"\nTraining for epoch {k+1} completed.\n")
        print(f"Time taken for one epoch: {epoch_duration} seconds")


        start_time_lag = time.time()

        global_step_list.append(global_step)
        total_batch_epoch_accuracy.append(sum(batch_epoch_accuracy)/len(batch_epoch_accuracy))
        total_batch_epoch_loss.append(sum(batch_epoch_loss)/len(batch_epoch_loss))



        predictions_validation_iteration=[]
        predictions_iteration=[]
        model.eval()


        # with torch.no_grad():
        #     for features in test_loaded_data:  #we do this because we load the data in batches, the features is the batches
        #         for i in range(0,len(features)):
        #             outputs = model(features[i])
        #             outputs = outputs.to(device)
        #             final_append=(((outputs * std_sales) + mean_sales).float().item())
        #             predictions_iteration.append(final_append)  # Collecting predictions
        # predictions_.append(predictions_iteration)
        with torch.no_grad():
            for features in validation_loaded_data:
                outputs = model(features[0])
                outputs=outputs.to(device)
                final_value=((outputs * std_validation) + mean_validation)
                for i in range(0,len(final_value)):
                    predictions_validation_iteration.append(final_value[i].item())  #we do this because of the way the tensors loaded into the data
                predictions_validation.append(predictions_validation_iteration)
        #--------------------------------------
        tensor_validaion_iteration = torch.tensor(predictions_validation_iteration).float()
        tensor_validation_sales = torch.tensor(validation_sales).float()

        mean_tensor_validaion_iteration = tensor_validaion_iteration.mean()
        std_tensor_validaion_iteration = tensor_validaion_iteration.std()
        mean_tensor_validation_sales = tensor_validation_sales.mean()
        std_tensor_validation_sales = tensor_validation_sales.std()

        normalized_tensor_validaion_iteration = ( tensor_validaion_iteration - mean_tensor_validation_sales) / std_tensor_validation_sales
        normalized_tensor_validation_sales = (  tensor_validation_sales - mean_tensor_validation_sales) / std_tensor_validation_sales

        loss_validation = loss_function(normalized_tensor_validaion_iteration, normalized_tensor_validation_sales)
        loss_set_iteration.append( loss_validation.item())


        for i in range(0, len(predictions_validation_iteration)):

            totalcount = 0
            workingcount = 0
            percent_within_lower_bound = (confidence) * validation_sales[i]
            percent_within_upper_bound = (2 - confidence) * validation_sales[i]

            workingcount += percent_within_lower_bound <= predictions_validation_iteration[i] <= percent_within_upper_bound
            totalcount += 1
            differance = predictions_validation_iteration[i] - validation_sales[i]
            percent_set_iteration.append(abs(100 * (differance / validation_sales[i])))   #these absolute differances will be more because of no normalization
            absolute_percent.append(workingcount / totalcount)









        average_percent_iteration = sum(percent_set_iteration) / len(percent_set_iteration)
        average_abs_percent_iteration = sum(absolute_percent) / len(absolute_percent)

        average_percent.append(average_percent_iteration)
        average_abs.append(average_abs_percent_iteration)



        writer.add_scalar('Validation/Absolute Accuracy', average_abs[k], global_step_loss[k])
        writer.add_scalar('Validation/Relative Error', average_percent[k], global_step_loss[k])
        writer.add_scalar('Validation/Loss: MSE', loss_set_iteration[k], global_step_loss[k])
        epoch_count += 1

        end_time_lag = time.time()
        epoch_duration_lag = end_time_lag - start_time_lag
        print(f"Time between epochs: {epoch_duration_lag} seconds\n-------------------------------------------")


    writer.close()



#we save the zip file here using the below code
    Path_model = os.path.join(model_path, f"exp{exp_number}_Model_Path")
    torch.save(model.state_dict(), Path_model)


    device = torch.device("cuda")
    model = Train_Model()
    model.load_state_dict(torch.load(Path_model, map_location="cuda:0"))  # Choose whatever GPU device number you want
    model.to(device)

    model.eval()
    print(model)

    # with torch.no_grad():
    #     for features in test_loaded_data:  #we do this because we load the data in batches, the features is the batches
    #         for i in range(0,len(features)):
    #             outputs = model(features[i])
    #             outputs = outputs.to(device)
    #             final_append=(((outputs * std_sales) + mean_sales).float().item())
    #             predictions_iteration.append(final_append)  # Collecting predictions
    # predictions_.append(predictions_iteration)
    #now, we would evaluate the model using test data


    # print(f"House Predictions of All Test_Data (largest epoch): \n{predictions_[-1][:shown_data]}\n")
    # print(f"House Predictions of number {test_id} Test_Data:\n{predictions_[-1][test_id-1]} \n")
    # print(f"Actual Values of Validation Set (Largest Epoch): \n{validation_sales[:shown_data]}\n")
    # print(f"House Predictions of Validation Set (Largest Epoch): \n{predictions_validation[-1][:shown_data]}\n")
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
