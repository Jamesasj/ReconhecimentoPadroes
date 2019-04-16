#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:58:04 2017

@author: thomas
"""

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from util import pause

def KNN( X, y, classname ):
    print ('Classifier: K-Nearest Neighbor\n')
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    #skf.get_n_splits(X, y)
    #print skf

    # instantiate learning model (k = 3)
    knn = KNeighborsClassifier(n_neighbors=3)

    y_pred_overall = []
    y_test_overall = []

    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        #print '\ny_test[0]=', y_test[0], '\ny_pred[0]=', y_pred[0]
        #pause()
        
        y_pred_overall = np.concatenate([y_pred_overall, y_pred])
        y_test_overall = np.concatenate([y_test_overall, y_test])

    '''
        print(classification_report(y_test, y_pred, target_names=classname))
        print(f1_score(y_test, y_pred, average='micro'))
        print(f1_score(y_test, y_pred, average='macro'))
        print(cv_cm)
    '''

    print ('KNN Classification Report: ')
    print (classification_report(y_test_overall, y_pred_overall, target_names=classname, digits=3))
    print ('Accuracy=', '%.2f %%' % (100*accuracy_score(y_test_overall, y_pred_overall)))
    print ('KNN Confusion Matrix: ')
    print (confusion_matrix(y_test_overall, y_pred_overall))


if __name__ == '__main__':

    import sklearn.datasets as datasets

    # Get iris data
    iris = datasets.load_iris()
    featname = iris['feature_names']
    X = iris.data
    y = iris.target
    classname = iris.target_names

    KNN( X, y, classname )
