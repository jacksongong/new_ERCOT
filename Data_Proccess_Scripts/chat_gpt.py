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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
class Linear_Regression:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = LinearRegression()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2


def run_linear_regression(data_path, test_size, random_state, epochs, exp_number, batch_size):
    ohe_path = os.path.join(data_path, "Hot_Encoded_Data_Final_DF.csv")
    concat_data = pd.read_csv(ohe_path)
    actual_desired_lambda = 'SystemLambda_ActualValues'
    before_sales_column = concat_data[actual_desired_lambda]
    concat_data = concat_data.drop(columns=[actual_desired_lambda])
    stat_model, stat_model_validation = train_test_split(concat_data, test_size=test_size, random_state=random_state)
    sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size, shuffle=False,
                                                      random_state=random_state)
    global_step = 0
    mse_list = []
    r2_list = []
    x_step = []
    writer = SummaryWriter(f"log/exp{exp_number}")

    num_batches = len(stat_model) // batch_size + (1 if len(stat_model) % batch_size != 0 else 0)

    for epoch in range(epochs):
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            X_batch = stat_model[start_idx:end_idx]
            y_batch = sales_column[start_idx:end_idx]

            model = Linear_Regression(X_batch, y_batch, stat_model_validation, validation_sales)
            model.train_model()

        mse, r2 = model.evaluate_model()
        writer.add_scalar('Validation/Absolute Accuracy', mse, global_step)
        writer.add_scalar('Validation/Relative Error', r2, global_step)
        mse_list.append(mse)
        r2_list.append(r2)
        global_step += 1

    for i in range(len(mse_list)):
        x_step.append(i + 1)
    plt.figure(figsize=(10, 5))
    plt.title('Loss and Accuracies During Training')
    plt.plot(x_step, r2_list, label='R2 During Training')
    plt.plot(x_step, mse_list, label='MSE Loss Training')
    plt.legend(title="Legend", loc='upper right', fontsize='x-small')
    plt.show()