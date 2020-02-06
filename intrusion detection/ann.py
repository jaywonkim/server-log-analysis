import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# fix random seed for reproducibility
np.random.seed(7)

train_data = pd.read_csv(
    "/home/akshaj/projects/playground_py3.5/server_log_analysis/intrusion detection/labeled-data-samples/all.csv")
# test_data = pd.read_csv(
#     "/home/akshaj/projects/playground_py3.5/server_log_analysis/intrusion detection/labeled-data-samples/oct-2016 (copy).csv")

X_train = train_data[["URL_LENGTH", "PARAM", "REPLY_CODE"]]
Y_train = train_data["RESULT"]
# X_test = test_data[["URL_LENGTH", "PARAM", "REPLY_CODE"]]
# Y_test = test_data["RESULT"]
# display(X_train)

X_train_encode = pd.get_dummies(X_train["REPLY_CODE"])
X_train = X_train.drop("REPLY_CODE", axis=1)
X_train = X_train.join(X_train_encode)

# X_test_encode = pd.get_dummies(X_test["REPLY_CODE"])
# X_test = X_test.drop("REPLY_CODE", axis=1)
# X_test = X_test.join(X_test_encode)

features_train = X_train.iloc[:, 0:8]
X = features_train.values
Y = Y_train.values

# features_test = X_test.iloc[:, 0:8]
# X_t = features_test.values
# Y_t = Y_test.values


X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, random_state=0, test_size=0.3)

sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)

classifier = Sequential()
classifier.add(Dense(units=12, kernel_initializer='uniform',
                     activation='relu', input_dim=8))
classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
classifier.add(
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling Neural Network
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting our model
classifier.fit(X_tr, Y_tr, batch_size=10, epochs=100)
scores = classifier.evaluate(X_te, Y_te)
print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1] * 100))

# Predicting the Test set results
y_pred = classifier.predict(X_te)
# y_pred = (y_pred > 0.5)
y_pred = [round(x[0]) for x in y_pred]

cm = confusion_matrix(Y_te, y_pred)
print("\nConfusion matrix: \n", cm)
