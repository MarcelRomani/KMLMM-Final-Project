import loading as l
import gridsearch as g
import scale as s
from kfdaM import Kfda
from GridKernelFDA import GridKernelFDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import make_scorer, roc_auc_score, recall_score, accuracy_score,confusion_matrix, precision_score, classification_report


data_path = './Data/pd_speech_features.csv'
X, y = l.load_data(data_path)


X_train, X_test, y_train, y_test = s.split_and_scale(X, y)


############################# Kernel FDA ###########################################
param = [{'kernel' : ['rbf', 'laplacian', 'chi2'],
         'gamma' : [None, 0.0001, 0.001, 0.01, 0.1, 1],
         'robustness_offset': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]},
         {'kernel' : ['linear', 'polynomial', 'sigmoid'],
         'gamma' : [None, 0.0001, 0.001, 0.01, 0.1, 1],
         'degree' : [2, 3, 4, 5, 6, 7, 8, 9, 10],
         'robustness_offset': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]}
]  
gsKernel = GridKernelFDA(X_train, y_train, param, metrics=['accuracy', 'f1', 'recall'])
best_params = gsKernel.get_best_params()

cls = Kfda(**best_params)
cls.fit(X_train, y_train)
preds = cls.predict(X_test)

reportFDA = classification_report(y_test, preds)
cm = confusion_matrix(y_test, preds)
print(reportFDA)
print(cm)
print(best_params)


############################# LDA ###########################################
clsLDA = LinearDiscriminantAnalysis(n_components=1)
clsLDA.fit(X_train, y_train)
predsLDA = clsLDA.predict(X_test)

reportLDA = classification_report(y_test, predsLDA)
cmLDA = confusion_matrix(y_test, predsLDA)
print(reportLDA)
print(cmLDA)
