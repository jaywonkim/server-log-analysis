import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
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


model = DecisionTreeClassifier()
model.fit(X_tr, Y_tr)
y_predict = model.predict(X_te)


clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
                                  max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_tr, Y_tr)
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                     max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_tr, Y_tr)


y_pred = clf_gini.predict(X_te)
y_pred_en = clf_entropy.predict(X_te)

# accuracy
print("\nAccuracy with criterion Default is " +
      str(accuracy_score(Y_te, y_predict) * 100))
print("\n")


print("\nAccuracy with criterion as gini index is " +
      str(accuracy_score(Y_te, y_pred) * 100))
print("\n")


print("\nAccuracy criterion as information gain is " +
      str(accuracy_score(Y_te, y_pred_en) * 100))

print("\n")
print("\nConfusion Matrix: ")
print(confusion_matrix(Y_te, y_predict))
# tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)
