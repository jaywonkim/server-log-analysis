# %%
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import sqrt
import datetime
import pylab


# %%
# assign server_logs as data frame
server_logs = pd.read_csv(
    "/home/akshaj/projects/playground_py3.5/server_log_analysis/gen_analysis/data/dataset2/access_train_sample.csv")
server_logs = server_logs.dropna()
# server_logs = server_logs
# convert TIME column to datetime format
server_logs['DATE_TIME'] = pd.to_datetime(
    server_logs.DATE_TIME, format='%d/%m/%Y %H:%M:%S', errors='coerce')


# convert REPLY_SIZE to numeric type
# server_logs.loc[server_logs["REPLY_SIZE"] == "-", "REPLY_SIZE"] = 0
# pd.to_numeric(server_logs["REPLY_SIZE"])

# Remove wrong rows from the data frame
server_logs = server_logs[server_logs['REPLY_CODE'].isin(
    ['200', '304', '302', '404', '501', '403', '500'])]

# server_logs.to_csv(
#     "/home/akshaj/projects/playground_py3.5/server_log_analysis/forecasting/data/dataset2/server_logs.csv")

# print(server_logs["TIME"])
# %%
# 1. Plot the percentage reply_codes. eg: number of HTTP 200 replies.
# reply_codes = server_logs["REPLY_CODE"].value_counts()
# # reply_codes[:10].plot(kind="pie")
# labels = [r'200', r'304', r'302', r'404', r'501', r'403', r'500']
# patches, texts = plt.pie(reply_codes[:10])
# plt.legend(patches, labels, loc="best")
# # plt.axis('equal')
# plt.show()

# %%
# 2. Plot server usage(bytes used/size of content returned by the web server) across the whole day.
# reply_size plotted with unique time
# v = server_logs.groupby('DATE_TIME')['REPLY_SIZE'].sum()
# a, b = v.index.tolist(), v.tolist()
# plt.plot(a, b)
# plt.xticks(rotation=90)
# plt.show()

# %%
# 3. reply_size is plotted with time_of_day
# time_of_day = server_logs.groupby(
#     "DATE_TIME")['REPLY_SIZE'].sum().plot(kind="bar")
# # c, d = time_of_day.index.tolist(), time_of_day.tolist()
# # plt.plot(c, d)
# plt.show()


#%%
# 4. Frequency by which unique hosts accessed the server. min/max/mean can be calculated and plotted.
# You can also make a seperate dataframe for this.
# Number of requests made by all uniques users
# user_freq_pair = server_logs["HOST"].head(100).value_counts()
# user, freq = user_freq_pair.index.tolist(), user_freq_pair.tolist()
# plt.bar(user, freq)
# plt.xticks(rotation=90)
# plt.show()


# %%
# Rank target urls by frequency at which they were accessed
# target_freq_pair = server_logs.head(100).copy()
# target_freq_pair["TARGET"] = target_freq_pair["TARGET"].astype("category")
# target_freq_pair["TARGET_CAT"] = target_freq_pair["TARGET"].cat.codes
# target_freq = target_freq_pair["TARGET_CAT"].value_counts()
# target, freq = target_freq_pair["TARGET_CAT"].index.tolist(
# ), target_freq_pair["TARGET_CAT"].tolist()
# plt.boxplot(freq, target)
# plt.show()


# %%
# series = server_logs.groupby('DATE-TIME')['REPLY_SIZE'].sum()
# values = series.values
# values = values.reshape((len(values), 1))
# scaler = StandardScaler()
# scaler = scaler.fit(values)
# plt.plot(a, scaler)
# plt.show()
# print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
# normalized = scaler.transform(values)
# inverse transform and print the first 5 rows
# inversed = scaler.inverse_transform(normalized)

# %%
###########################################################################
# Use one hot encodeing for the unique user IPs                           #
# calculate the mean of reply_size of all the unique users                #
# If anyone user has a reply_size of more than 2 times the mean at a      #
# single point in time, they are an anamoly.                              #
# Use logistic regression/ neural network for it                          #
# DO THE SAME FOR UNIQUE IP AND FREQUENCY OF REQUESTS TO PLOT THE ANOMALY #
###########################################################################

############################################################################
# FEATURE EXTRACTION
#########################################
# 1.SESSION CREATION

# create_sess = server_logs.copy()
# # METHOD 1
# # create_sess = create_sess.sort_values(by=[‘HOST’, ’USER_AGENT’])
# g = create_sess.groupby(["HOST", "USER_AGENT"])
# create_sess["SESSION_ID"] = g["DATE_TIME"].apply(lambda s: (
#     s - s.shift(1) > pd.Timedelta(minutes=30)).fillna(0).cumsum(skipna=False))
#
#
# print(server_logs[server_logs["HTTP_METHOD"].isin(["HEAD"])])


# METHOD 2
# gt_15min = session_creation.DATE_TIME.diff() > pd.datetools.timedelta(minutes=30)
# diff_user = session_creation.HOST != session_creation.HOST.shift()
# session_id = (diff_user | gt_15min).cumsum()
# df['session_id'] = session_id.map(pd.Series(list(ascii_uppercase)))
# print(session_creation)


############################################
# UNIQUE TIME VS REPLY_SIZE TO PREDICT TRAFFIC
# v = server_logs.groupby('TIME')['REPLY_SIZE'].sum()
# a, b = v.index.tolist(), v.tolist()


# ax = v.plot()
# ax = plt.(a, b)
# ax.set_xlim(pd.Timestamp("2017-12-30 00:00:02"),
#             pd.Timestamp("2017-12-30 23:53:07"))
# plt.show()
