import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader #here, the pytorch ALREADY HAS pre-defined functions that can be used through the Dataset and DataLoader functions.
from sklearn.model_selection import train_test_split
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse
import pathlib
from torchvision.utils import make_grid
import datetime
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



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

args = parser.parse_args()
data_path = args.data_path
name_csv = args.name_csv

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
df['DateTime'] = df['DateTime'].dt.strftime('%m/%d')
ohe_all_data = pd.get_dummies(df['DateTime'])
ohe_all_data = ohe_all_data.astype(bool).astype(int)



train_drop = df.drop(columns=[common_delivery, common_time, actual_desired_lambda, 'DateTime'])


train_drop = train_drop.iloc[:, 1:]
before_numbers_train = train_drop.select_dtypes(exclude=['object'])

before_numbers_train.fillna(0, inplace=True)


before_split_train_manipulated = before_numbers_train.apply(lambda x: (x - x.mean()) / (x.std()))


concat_data = pd.concat([ohe_all_data, before_split_train_manipulated], axis = 1)    #we CANNOT normalize dummy variables


print(concat_data)



stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size, random_state=random_state)
sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size, random_state=random_state)


input = stat_model.shape[0]
attibutes = stat_model.shape[1]



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


train_features = torch.tensor(stat_model.values).float()
#test_features = torch.tensor(stat_model_test.values).float()
validation_features = torch.tensor(stat_model_validation.values).float()

sales_column_tensor = torch.tensor(sales_column.values).float()


validation_sales_tensor=torch.tensor(validation_sales.values).float()
mean_validation=validation_sales_tensor.mean()
std_validation=validation_sales_tensor.std()
validation_normalized_sales = (validation_sales_tensor-mean_validation)/std_validation


mean_sales=sales_column_tensor.mean()
std_sales=sales_column_tensor.std()
normalized_sales = (sales_column_tensor-mean_sales)/std_sales

normalized_tensor_sales = torch.tensor(normalized_sales).float()

train_dataset_custom_dataset = CustomTensorDataset(train_features, normalized_tensor_sales)
#test_dataset_custom_dataset = CustomTestDataset(test_features)
validation_dataset_custom_dataset = CustomTensorDataset(validation_features, validation_normalized_sales)

train_loaded_data = DataLoader(train_dataset_custom_dataset, batch_size=batch_size, shuffle=True)
validation_loaded_data = DataLoader(validation_dataset_custom_dataset, batch_size=batch_size, shuffle=False)
#test_loaded_data = DataLoader(test_dataset_custom_dataset, batch_size=batch_size, shuffle=False)

class Train_Model(torch.nn.Module):
    def __init__(self):
        super(Train_Model, self).__init__()

        self.attibutes = torch.tensor(attibutes, dtype=torch.float).to(device)


        # Define layers correctly without trailing commas
        self.layer1 = nn.Linear(attibutes, 1024)
        self.dropout1 = nn.Dropout(p=0.4)
        self.layer2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(p=0.4)
        self.layer3 = nn.Linear(1024, 1)
        #self.dropout3 = nn.Dropout(p=0.5)
        #self.layer4 = nn.Linear(1024, 1024)
        #self.dropout4 = nn.Dropout(p=0.5)
        #self.layer5 = nn.Linear(1024, 1)
        self.model_version = nn.PReLU()
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        #nn.init.xavier_uniform_(self.layer4.weight)
        #nn.init.xavier_uniform_(self.layer5.weight)
    def forward(self, x):
        x = self.model_version(self.layer1(x))
        x = self.dropout1(x)
        x = self.model_version(self.layer2(x))
        x = self.dropout2(x)
        x = self.model_version(self.layer3(x))
        #x = self.dropout3(x)
        #x = self.model_version(self.layer4(x))
        #x = self.dropout4(x)
        #x = self.model_version(self.layer5(x))

        return x.squeeze()
# ----------------------------------------------------------------------------------------

loss_function=nn.MSELoss()

def train_one_epoch(model, optimizer, data_loader,global_step):
    model.train().to(device)
    accuracy_relative=[]
    loss_list = []

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

        accurate_reform=[]
        for i in range(0, len(accuracy_percent)):
            accurate_reform.append(accuracy_percent[i].item())
        mean_accuracy=sum(accurate_reform)/len(accurate_reform)

        loss_list.append(loss.item())
        accuracy_relative.append(mean_accuracy)

        writer.add_scalar('Training/Relative Percent Error Over Time', mean_accuracy, global_step)
        global_step+=1

    return(sum(loss_list)/len(loss_list)), accuracy_relative, loss_list

def train_all(model, optimizer, loaded_data, global_step, iteration, list):
    list_=list
    model_final_epoch = []
    loss_after, batch_accuracy, batch_loss_list = train_one_epoch(model, optimizer, loaded_data,global_step)
    model_final_epoch.append(loss_after)
    print(f"Train Epoch Number: {list_[iteration]}, Loss Average for ALL batches: {loss_after}")
    tensor_model_total_batch_loss = torch.tensor(model_final_epoch).float()
    return(tensor_model_total_batch_loss),batch_accuracy, batch_loss_list



if __name__ == '__main__':
    writer = SummaryWriter("log/exp25")
    global_step = 1

    epochs = 200

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
    weight_decay = 1e-2

    total_batch_epoch_accuracy=[]
    #total_batch_epoch_accuracy_abs=[]
    total_batch_epoch_loss=[]
    absolute_percent=[]

    # learning_rate = 1e-5

    if epoch_count<=5:
        learning_rate = 1e-5
    else:
        learning_rate=pow(1,-epoch_count)

    model = Train_Model().to(device)
    for k in range (0, epochs):  #this is already training the model 4 times, and we are training one epoch functions 4 times as well in the
        print(f"Training the {k+1} epoch:\n")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #print(epoch_list)
        total_batch_loss,batch_epoch_accuracy, batch_epoch_loss=train_all(model, optimizer, train_loaded_data, global_step,k,epoch_list)
        print(f"\nTraining for epoch {k+1} completed.\n-------------------------------------------")

        total_batch_epoch_accuracy.append(sum(batch_epoch_accuracy)/len(batch_epoch_accuracy))
        total_batch_epoch_loss.append(sum(batch_epoch_loss)/len(batch_epoch_loss))



        predictions_validation_iteration=[]
        predictions_iteration=[]  #this will give the predictions based on the epoch that we have trained already ( we append this as a list into the predictions_ global list

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
                #print(final_value[1])
                for i in range(0,len(final_value)):
                    predictions_validation_iteration.append(final_value[i].item())  #we do this because of the way the tensors loaded into the data
                predictions_validation.append(predictions_validation_iteration)
        #--------------------------------------
        #print(f"ssssssssssssssssssssssssssssss{predictions_validation_iteration}")

        tensor_validaion_iteration = torch.tensor(predictions_validation_iteration).float()
        tensor_validation_sales = torch.tensor(validation_sales.tolist()).float()

        mean_tensor_validaion_iteration = tensor_validaion_iteration.mean()
        std_tensor_validaion_iteration = tensor_validaion_iteration.std()
        mean_tensor_validation_sales = tensor_validation_sales.mean()
        std_tensor_validation_sales = tensor_validation_sales.std()

        # the two variables:  normalized_tensor_validaion_iteration,normalized_tensor_validation_sales      are simply the normalized versions of what we already have .

        # here, we should normalize by taking the average of the prediction values... NOT BY USNIG THE MEAN OF THE ACTUAL VALUES
        normalized_tensor_validaion_iteration = ( tensor_validaion_iteration - mean_tensor_validation_sales) / std_tensor_validation_sales
        normalized_tensor_validation_sales = (  tensor_validation_sales - mean_tensor_validation_sales) / std_tensor_validation_sales

        # these are simply the normalized, tensored, values of what we have. GOOD.   #at some point, the normalized predictions and actual values become very close (
        loss_validation = loss_function(normalized_tensor_validaion_iteration, normalized_tensor_validation_sales)
        loss_set_iteration.append( loss_validation.item())  # this gives the loss that we want for each iteration, we must turn this into a tensor for each value and graph it

        # we can create a list here,

        # normalized_tensor_validaion_iteration,normalized_tensor_validation_sales
        # print(normalized_tensor_validaion_iteration)
        # print(len(normalized_tensor_validaion_iteration))
        # print(normalized_tensor_validation_sales)
        # print(len(normalized_tensor_validation_sales))

        # take the percent differance from scaled values, not normalized ones.
        # scaled_validate_prediction=normalized_tensor_validaion_iteration*  +mean_tensor_validation_sales
        for i in range(0, len(predictions_validation_iteration)):  # there are 460 items in the validation set for the comparison in the KNOWN TARGET VALUES after split
            totalcount = 0
            workingcount = 0
            # here, we can take within 95% confidence from the actual sales
            # print(validation_sales.tolist())
            percent_within_lower_bound = (confidence) * validation_sales.tolist()[i]
            percent_within_upper_bound = (2 - confidence) * validation_sales.tolist()[i]

            workingcount += percent_within_lower_bound <= predictions_validation_iteration[i] <= percent_within_upper_bound

            totalcount += 1
            # print(predictions_validation_iteration[i])
            differance = predictions_validation_iteration[i] - validation_sales.tolist()[i]
            percent_set_iteration.append(abs(100 * (differance / validation_sales.tolist()[i])))  # using relative percent error as (differance prediction and actual)/actual-----> this will create a 460 term list
            absolute_percent.append(workingcount / totalcount)

        average_percent_iteration = sum(percent_set_iteration) / len(percent_set_iteration)
        average_abs_percent_iteration = sum(absolute_percent) / len(absolute_percent)

        average_percent.append(average_percent_iteration)  # this will be the relative error in the validation set average over all the predictions and this average is the relative accuracy for the set
        average_abs.append(average_abs_percent_iteration)
        # print(f"dddddddddddddddddddddddddddddd{average_percent[k]}")
        # print(loss_set_iteration[k])   #this value is NOT DECREASING
        #

        # print(average_percent_iteration)
        # print(loss_validation)

        # why are we only getting the last value here?
        writer.add_scalar('Validation/Absolute Accuracy', average_abs[k],
                          global_step)  # the scalar would be in terms of TIME   #here, we have already had the loss value of each batch recorded. WE DO NOT TAKE THE AVERAGE

        writer.add_scalar('Validation/Relative Error', average_percent[k],
                          global_step)  # the scalar would be in terms of TIME   #here, we have already had the loss value of each batch recorded. WE DO NOT TAKE THE AVERAGE
        # writer.flush()
        writer.add_scalar('Validation/Loss: MSE', loss_set_iteration[k],
                          global_step)  # the scalar would be in terms of TIME
        # writer.flush()
        global_step += 1
        epoch_count += 1
    writer.close()

    # print(f"House Predictions of All Test_Data (largest epoch): \n{predictions_[-1][:shown_data]}\n")
    # print(f"House Predictions of number {test_id} Test_Data:\n{predictions_[-1][test_id-1]} \n")
    # print(f"Actual Values of Validation Set (Largest Epoch): \n{validation_sales.tolist()[:shown_data]}\n")
    # print(f"House Predictions of Validation Set (Largest Epoch): \n{predictions_validation[-1][:shown_data]}\n")
    print(f"Loss between actual and predicted validation set (Largest Epoch): \n{loss_set_iteration}\n")   #this will be the loss for each of 4 epochs GRAPH THIS
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


    # print(total_batch_epoch_loss)
    # print(total_batch_epoch_accuracy)



    # print(f"ddddddddddddddddddddddddddddddddddd{average_percent}")
    # print(f"{loss_set_iteration}")

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

    #print(min(loss_set_iteration))
    plt.legend(title="Legend", loc='upper right', fontsize='x-small')
    plt.show()
    # print(average_percent)
    # print(loss_set_iteration)