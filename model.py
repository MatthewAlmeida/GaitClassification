from __future__ import division

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import sklearn.ensemble as ens


if __name__ == "__main__":
    X = np.load("./data/X.npy")
    y = np.load("./data/y.npy")

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.33)
    scaler = StandardScaler()
    gbst = ens.GradientBoostingClassifier(learning_rate=0.001, n_estimators=5000)

    fold = 0

    for train_index, test_index in sss.split(X, y):
        X_Train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_Train_Scaled = np.copy(X_Train)
        X_Test_Scaled = np.copy(X_test)

        X_Train_Scaled[:,8:] = scaler.fit_transform(X_Train[:,8:])
        X_Test_Scaled[:,8:] = scaler.transform(X_test[:,8:])

        gbst.fit(X_Train_Scaled, y_train)

        train_acc = gbst.score(X_Train_Scaled, y_train)
        test_acc = gbst.score(X_Test_Scaled, y_test)

        print("Fold: {0}".format(fold))
        print("  Accuracy vs training set: {0}".format(train_acc))
        print("  Accuracy vs testing set: {0}".format(test_acc))

        fold += 1
