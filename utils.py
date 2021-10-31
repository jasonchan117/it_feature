import nltk
import numpy as np
import pandas as pd
import string
from nltk.parse.stanford import StanfordDependencyParser
import spacy
import re
from nltk.corpus import wordnet as wn
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