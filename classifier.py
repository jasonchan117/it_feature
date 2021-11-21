import numpy as np
import pandas as pd
import numpy as np
from collections import Counter
from utils import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import  RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import sys
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer

kf = KFold( n_splits=10, shuffle=True, random_state=10)

x, y = process_raw_data('results.csv') # NomAnaph = 1
DT_model = DecisionTreeClassifier()
LR_model = LogisticRegression()
RF_model = RandomForestClassifier()
SVM_model = SVC()
#{1.0: 450, 0.0: 96}
'''
2) Apply ten fold
'''
acc = train_and_val(x = x, y = y, kf = kf, DT_model =  DT_model, LR_model = LR_model, RF_model = RF_model, SVM_model = SVM_model)
print('The best results of each model:')
print(acc)

'''
3) Use SMOTE on the dataset
'''
smo = SMOTE(random_state=42)
x_smo, y_smo = smo.fit_resample(x, y)
acc = train_and_val(x = x_smo, y = y_smo, kf = kf, DT_model =  DT_model, LR_model = LR_model, RF_model = RF_model, SVM_model = SVM_model)
print('The best results of each model after SMOTE:')
print(acc)
'''
4) Hyper-parameters fine-tuning
'''
# Decision Tree
C = [0.001, 0.002, 0.003, 0.004, 0.005]
M = [2, 3, 4, 5, 6, 7, 8, 9, 10]
DT_param_set = []
for i in C:
    for j in M:
        DT_param_set.append([i, j])
max_DT = -1
param_DT = []

for i in DT_param_set:

    DT_model = DecisionTreeClassifier(ccp_alpha=i[0], min_samples_split=i[1])
    acc = train_and_val(x = x, y = y, kf = kf, DT_model =  DT_model)['DT_model']

    if acc > max_DT:
        max_DT = acc
        param_DT = i
print('Decision Tree -> Accuracy:{}, Parameters C:{}, M:{}'.format(max_DT, param_DT[0], param_DT[1]))


# Random Forest
max_RF = -1
param_RF = []
I = [i for i in range(100, 1001, 100)]
K = [i for i in range(5, 28, 2)]

RF_param_set = []
for i in I:
    for j in K:
        RF_param_set.append([i, j])

for i in RF_param_set:
    RF_model = RandomForestClassifier(n_estimators=i[0], max_features=i[1])
    acc = train_and_val(x=x_smo, y=y_smo, kf=kf, RF_model=RF_model)['RF_model']
    print(acc)
    if acc > max_RF:
        max_RF = acc
        param_RF = i
print('Random Forest -> Accuracy:{}, Parameters I:{}, K:{}'.format(max_RF, param_RF[0], param_RF[1]))


# Logistic Regression
max_LR = -1
param_LR = []
ridge = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e10, 1e9, 1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1]
for i in ridge:
    LR_model = LogisticRegression(C = i)
    acc = train_and_val(x=x_smo, y=y_smo, kf=kf, LR_model=LR_model)['LR_model']
    print('-->', acc)
    if acc > max_LR:
        max_LR = acc
        param_LR = i
print('Logistic Regression -> Accuracy:{}, Parameters ridge:{}'.format(max_LR, param_LR))

# SVM
max_SVM = -1
param_SVM = []
cost = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
SVM_param_set = []
for i in cost:
    for j in cost:
        SVM_param_set.append([i, j])

for i in SVM_param_set:
    SVM_model = SVC(gamma=i[1], C = i[0])
    acc = train_and_val(x=x, y=y, kf=kf, SVM_model=SVM_model)['SVM_model']
    if acc > max_SVM:
        max_SVM = acc
        param_SVM = i
print('SVM -> Accuracy:{}, Parameters C:{}, gamma:{}'.format(max_SVM, param_SVM[0], param_SVM[1]))
'''
Feature Ranking
'''
res =  mutual_info_classif(x_smo, y_smo)
res = [0.21819637, 0.26307731,0.02089729,0.10408425,0.11165286,0.00157417
, 0.18526162, 0.20435355, 0.22257146, 0.21166985, 0.16536938, 0.22514492,
 0.18096379, 0.17603666, 0.02876315, 0.14915057, 0.10985039, 0.03965201,
 0.07511509, 0.0803626,  0.06559521, 0.0218448,  0.15778717, 0.,
 0.13496679, 0.01078187, 0.04066395]
features_name = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7_1', 'F7_2', 'F7_3', 'F7_4', 'F7_5' , 'F7_6' , 'F7_7', 'F7_8', 'F8','F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20']
features = []
for ind, i in enumerate(features_name):
    features.append([i, res[ind]])

features = sorted(features, key= lambda x:x[1], reverse= True)
for i in features:
    print(i[0], i[1])