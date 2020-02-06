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
# sc = StandardScaler()
# X_tr = sc.fit_transform(X_train)
# X_te = sc.fit(X_test)

logreg = LogisticRegression()
logreg.fit(X_tr, Y_tr)
print("\nThe precision of the Logistic Regression Classifier is: " +
      str(logreg.score(X_te, Y_te) * 100) + "%\n")

y_pred = logreg.predict(X_te)

print('\nAccuracy of logistic regression classifier on test set: {:.2f}'.format(
    logreg.score(X_te, Y_te)) + "\n")


kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = 'accuracy'
results = model_selection.cross_val_score(
    logreg, X_tr, Y_tr, cv=kfold, scoring=scoring)

print("\n10-fold cross validation average accuracy: %.3f" %
      (results.mean()) + "\n")

confusion_matrix = confusion_matrix(Y_te, y_pred)
print("\n")
print("Confusion Matrix: ")
print(confusion_matrix)
print("\n")
print(classification_report(Y_te, y_pred))
print("\n")

logit_roc_auc = roc_auc_score(Y_te, logreg.predict(X_te))
fpr, tpr, thresholds = roc_curve(Y_te, logreg.predict_proba(X_te)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
