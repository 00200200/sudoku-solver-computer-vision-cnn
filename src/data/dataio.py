# Import libraries and classes required for this example:
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np

def separate_xy(dataframe):
    # Separate the features and target data (they are typically saved together
    # to file, so this may be necessary to accompany data loading)
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, 4].values
    return [X, y]

def combine_xy(X,y):
    # Combine the features and target data into a single array that can be
    # saved to file if needed
    return np.concatenate((X,y[:, np.newaxis]),axis=1)

def load(datapath):
    # Convert dataset to a pandas dataframe:
    dataset = pd.read_csv(datapath, header=0) 
    [X, y] = separate_xy(dataset)
    return [X, y]

def save(X,y,savepath):
    # Place the data into a single dataframe and save it to file
    combined = combine_xy(X,y)
    df = pd.DataFrame(combined)
    df.to_csv(savepath, header=False, index=False)