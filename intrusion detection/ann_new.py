import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from IPython.display import display
dataset = pd.read_csv(
    "/home/akshaj/projects/playground_py3.5/server_log_analysis/intrusion detection/labeled-data-samples/jan-2017 (copy).csv")

X = dataset.iloc[:, 0:3].values
Y = dataset.iloc[:, 3].values
display(X)
label_encoder_X = LabelEncoder()
X[:, 2] = label_encoder_X.fit_transform(X[:, 2])

display(X)
onehotencoder = OneHotEncoder(categorical_features=[2])
X = onehotencoder.fit_transform(X).toarray()
# X = X[:, 1:]
display(X)
