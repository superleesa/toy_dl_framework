import numpy as np
import pandas as pd

def normalize_pixels(X):
    normalized_X = X / 255
    return normalized_X

def change_to_one_hot_encode(y):
    all_data = []
    for correct_idx in y:
        one_hot = np.array([0] * 10)
        one_hot[correct_idx] = 1
        all_data.append(one_hot)

    Y = np.stack(all_data)
    return Y

def import_data_as_df():
    df1 = pd.read_csv("mnist_train.csv")
    df2 = pd.read_csv("mnist_test.csv")
    return df1, df2

def process(df1, df2):
    # separate the features and labels
    X_train = df1.loc[:,df1.columns != "label"]
    y_train = df1.loc[:, "label"]

    X_test = df2.loc[:, df2.columns != "label"]
    y_test = df2.loc[:, "label"]

    # normalize pixels
    X_train = normalize_pixels(X_train).to_numpy()
    X_test = normalize_pixels(X_test).to_numpy()

    # change to one-hot encode
    Y_train = change_to_one_hot_encode(y_train)
    Y_test = change_to_one_hot_encode(y_test)

    return X_train, Y_train, X_test, Y_test

def get_normalized_data():
    df1, df2 = import_data_as_df()
    return process(df1, df2)
