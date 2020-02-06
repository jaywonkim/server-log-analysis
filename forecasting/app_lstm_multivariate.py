# %%
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import sqrt
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = server_logs(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# %%
# assign server_logs as data frame
server_logs = pd.read_csv(
    "/home/akshaj/projects/playground_py3.5/server_log_analysis/forecasting/data/dataset1/clean_csv_logs.csv")

server_logs.index.name = "DATE_TIME"

values = server_logs.values
values = values[:, [3, 5, 6]]
print(values)
http_method = np.array(values[:, 0])
reply_code = np.array(values[:, 1])


label_encoder = LabelEncoder()
values[:, 0] = label_encoder.fit_transform(http_method)
values[:, 1] = label_encoder.fit_transform(reply_code)

# binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
#
# http_method_integer_encoded = http_method_integer_encoded.reshape(
#     len(http_method_integer_encoded), 1)
# http_method_onehot_encoded = onehot_encoder.fit_transform(
#     http_method_integer_encoded)
#
# reply_code_integer_encoded = reply_code_integer_encoded.reshape(
#     len(http_method_integer_encoded), 1)
# reply_code_onehot_encoded = onehot_encoder.fit_transform(
#     http_method_integer_encoded)


values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(
    reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
print(reframed.head())
# groups = [3, 5, 6]
# i = 1
# plt.figure()
# for group in groups:
#     plt.subplot(len(groups), 1, i)
#     plt.plot(values[:, group])
#     plt.title(server_logs.columns[group], y=0.5, loc='right')
#     i += 1
#
# plt.show()
