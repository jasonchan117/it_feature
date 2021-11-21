import nltk
import numpy as np
import pandas as pd
import string
from nltk.parse.stanford import StanfordDependencyParser
import spacy
import re
from nltk.corpus import wordnet as wn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
def is_atomicNP(tree):
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree and subtree.label() == 'NP':
            return False
    return True

def find_NP(tree, count = 0):
    if tree.label() == 'NP' and is_atomicNP(tree):
        count += 1
    new = 0
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            new = find_NP(subtree, count)
    return max(count, new)
def PP_follow(tree):
    # for subtree in tree:
        # if tree.label() == 'PP':
        #     return True
    if len(tree) == 0:
        return False
    if type(tree[len(tree) - 1]) == nltk.tree.Tree and tree[len(tree) - 1].label() == 'PP':
        return True
    return False

def NP_JJ_follow(tree):
    flag1 = False
    flag2 = False
    subflag = False
    if tree.label() == 'NP':
        flag1 = True

    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            subflag = NP_JJ_follow(subtree)
        else:
            if subtree[1] == 'JJ':
                flag2 = True
    return (flag1 and flag2) or subflag

def find_IV(tree):
    count = 0
    flag = 0
    flag2 = 0
    for sub in tree:
        count += 1
        if flag == 1 and type(sub) == nltk.tree.Tree and sub.label() == 'IV':
            count -= 1
            flag2 = 1
            break
        if type(sub) != nltk.tree.Tree and sub[0] == 'it':
            flag = 1
    return count if flag2 == 1 else 0
def count_tokens_PP(tree):
    count = 0
    flag = 0
    for sub in tree:
        if  sub[1]== 'IN':
            flag = 1
            break
        count += 1
    return count - 1 if flag == 1 else  0
def find_ADJ_NP(tree):
    if tree.label() == 'IN':
        pass
    ans = False
    under = False
    for i in range(len(tree)-1):
        if type(tree[i]) == nltk.tree.Tree:
            under = find_ADJ_NP(tree[i])
            if under == True:
                break
        if type(tree[i]) != nltk.tree.Tree and tree[i][1] == 'JJ' and type(tree[i+1]) == nltk.tree.Tree and tree[i+1].label() == 'NP':
            ans = True
            break

    return ans or under

def process_raw_data( csv = 'results.csv'):
    df = pd.read_csv(csv)
    labels = np.array(df.Labels == 'NomAnaph').astype('float32')
    F1 = np.array(df.F1).reshape(546, 1).astype('float32')
    F2 = np.array(df.F2).reshape(546, 1).astype('float32')
    F3 = np.array(df.F3).reshape(546, 1).astype('float32')
    F4 = np.array(df.F4).reshape(546, 1).astype('float32')
    F5 = np.array(df.F5).reshape(546, 1).astype('float32')
    F6 = np.array(df.F6).reshape(546, 1).astype('int').astype('float32')
    F7_1 = np.array(df.F7_1).reshape(546, 1)
    F7_2 = np.array(df.F7_2).reshape(546, 1)
    F7_3 = np.array(df.F7_3).reshape(546, 1)
    F7_4 = np.array(df.F7_4).reshape(546, 1)
    F7_5 = np.array(df.F7_5).reshape(546, 1)
    F7_6 = np.array(df.F7_6).reshape(546, 1)
    F7_7 = np.array(df.F7_7).reshape(546, 1)
    F7_8 = np.array(df.F7_8).reshape(546, 1)
    # {'JJS': 36, 'FW': 35, 'POS': 34, 'CD': 33, 'PDT': 32, 'RBR': 31, "''": 30, 'UH': 29, 'VBG': 28, 'RBS': 27, 'JJR': 26, 'WDT': 25, 'EX': 24, 'PRP$': 23, 'NNS': 22, 'RP': 21, 'NNP': 20, 'VBD': 19, 'TO': 18, 'PRP': 17, ':': 16, 'VBP': 15, 'VBN': 14, 'WP': 13, '.': 12, 'VB': 11, 'MD': 10, 'WRB': 9, 'CC': 8, 'NN': 7, 'JJ': 6, 'RB': 5, 'VBZ': 4, ',': 3, 'DT': 2, 'IN': 1, 'ABS': 0}
    F_7 = np.concatenate((F7_1, F7_2, F7_3, F7_4, F7_5, F7_6, F7_7, F7_8), axis=1)
    F7_dict = Counter(F_7.reshape(-1))
    for ind, i in enumerate(F7_dict.keys()):
        F7_dict[i] = ind
    for i in range(len(F_7)):
        for j in range(len(F_7[i])):
            F_7[i][j] = F7_dict[F_7[i][j]]
    F_7 = F_7.astype('float32')
    F8 = np.array(df.F8).reshape(546, 1).astype('int').astype('float32')
    F9 = np.array(df.F9).reshape(546, 1).astype('int').astype('float32')
    F10 = np.array(df.F10).reshape(546, 1).astype('float32')
    F11 = np.array(df.F11).reshape(546, 1).astype('int').astype('float32')
    F12 = np.array(df.F12).reshape(546, 1).astype('int').astype('float32')
    F13 = np.array(df.F13).reshape(546, 1).astype('int').astype('float32')
    F14 = np.array(df.F14).reshape(546, 1).astype('int').astype('float32')
    F15 = np.array(df.F15).reshape(546, 1).astype('float32')
    F16 = np.array(df.F16).reshape(546, 1).astype('float32')
    F17 = np.array(df.F17).reshape(546, 1).astype('int').astype('float32')
    F18 = np.array(df.F18).reshape(546, 1)
    F18_dict = Counter(F18.reshape(-1))
    # {'nsubj': 208, 'dobj': 208, 'pobj': 59, 'nsubjpass': 52, 'punct': 4, 'ROOT': 3, 'advcl': 2, 'attr': 2, 'cc': 2, 'nmod': 1, 'acomp': 1, 'neg': 1, 'advmod': 1, 'aux': 1, 'prep': 1}
    for ind, i in enumerate(F18_dict.keys()):
        F18_dict[i] = ind
    for i in range(len(F18)):
        for j in range(len(F18[i])):
            F18[i][j] = F18_dict[F18[i][j]]
    F18 = F18.astype('float32')
    F19 = np.array(df.F19).reshape(546, 1).astype('int').astype('float32')
    F20 = np.array(df.F20).reshape(546, 1).astype('int').astype('float32')

    x = np.concatenate((F1, F2, F3, F4, F5, F6, F_7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20), axis=1)

    return x, labels

def train_and_val( **param ):

    accuracy_set = {'DT_model': 0., 'LR_model': 0., 'RF_model': 0., 'SVM_model': 0.}

    for ind, (train_index, test_index) in enumerate(param['kf'].split(param['x'])):
        # print('Kfold Index -->', ind + 1)
        x_train = param['x'][train_index]
        x_test =param['x'][test_index]

        y_train = param['y'][train_index]
        y_test = param['y'][test_index]
        for i in param.keys():
            if i in {'DT_model', 'LR_model', 'RF_model', 'SVM_model'}:
                model = param[i].fit(x_train, y_train)
                y_pred = model.predict(x_test)
                # print(i + ':', accuracy_score(y_test, y_pred))
                accuracy_set[i] += accuracy_score(y_test, y_pred)

    for i in accuracy_set.keys():
        accuracy_set[i] /= 10.
    return accuracy_set