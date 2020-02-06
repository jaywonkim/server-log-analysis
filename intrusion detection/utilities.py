import sys
import numpy as np
from sklearn import tree, linear_model

traning_data = "/home/akshaj/projects/playground_py3.5/Intrusion-and-anomaly-detection-with-machine-learning-master/labeled-data-samples/jan-2017.csv"
testing_data = "/home/akshaj/projects/playground_py3.5/Intrusion-and-anomaly-detection-with-machine-learning-master/labeled-data-samples/oct-2016.csv"


def get_data_details(csv_data):
    data = np.genfromtxt(csv_data, delimiter=",")
    features = data[:, [0, 1, 2]]
    labels = data[:, 3]
    return features, labels


def get_occuracy(real_labels, predicted_labels, fltr):
    real_label_count = 0.0
    predicted_label_count = 0.0

    for real_label in real_labels:
        if real_label == fltr:
            real_label_count += 1

    for predicted_label in predicted_labels:
        if predicted_label == fltr:
            predicted_label_count += 1

    print("Real number of attacks:" + str(real_label_count))
    print("Predicted number of attacks:" + str(predicted_label_count))

    precision = predicted_label_count * 100 / real_label_count
    return precision
