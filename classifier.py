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
from sklearn.ensemble import  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
kf = KFold( n_splits=10, shuffle=True, random_state=10)

x, y = process_raw_data('results.csv')

DT_model = DecisionTreeClassifier()
LR_model = LogisticRegression()
RF_model = RandomForestClassifier()
SVM_model = LinearSVC
for train_index, test_index in kf.split(x):

    x_train = x[train_index]
    x_test = x[test_index]

    y_train = y[train_index]
    y_test = y[test_index]



