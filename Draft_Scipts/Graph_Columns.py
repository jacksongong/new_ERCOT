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

import os



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

common_time = 'Hour_Ending'
common_delivery = 'DeliveryDate'
actual_desired_lambda = 'SystemLambda_ActualValues'


# df_test_function = os.path.join(data_path, 'test_house.csv')
# df_function = os.path.join(data_path, 'train_house.csv')

df = pd.read_csv(df_function)

before_sales_column = df[actual_desired_lambda]
train_drop = df.drop(columns=[common_time, common_delivery, actual_desired_lambda])

#test_drop = df_test.drop(columns=['Id'])
before_numbers_train = train_drop.select_dtypes(exclude = ['object'])

before_numbers_train = before_numbers_train.iloc[:, 1:]


column_list = []
column_name = []
for cols in before_numbers_train.columns:
    column_list.append(before_numbers_train[cols])
    column_name.append(cols)

x_axis = [i for i in range (0, before_numbers_train.shape[0])]

writer = SummaryWriter("Graph_Columns/exp2")

for i in range (0, len(column_name)):
    local_step = 1

    fig, ax = plt.subplots()
    plt.title(f'{column_name[i]} Over Time')
    column_list = [element for element in before_numbers_train[column_name[i]]]
    plt.plot(x_axis, column_list, label='2018-(2024 June 18th)')
    plt.legend(title="Legend", loc='upper right', fontsize='x-small')

    ax.set_xlabel('Date and Time Span- 8760 per year')
    ax.set_ylabel('Y Value')

    for element in column_list:
        writer.add_scalar(f'Plots_Data/{column_name[i]}', element, local_step)
        local_step +=1
    #writer.flush()

    # global_step += 1

    #plt.show()

print(len(column_name))


