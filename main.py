import nltk
import numpy as np
import pandas as pd
import string
from nltk.parse.stanford import StanfordDependencyParser
import spacy
import re
from nltk.corpus import wordnet as wn
from utils import *
from spacy import displacy
from collections import Counter

from spacy_wordnet.wordnet_annotator import WordnetAnnotator

# a = "It does not concern you in the least."
# print(nltk.word_tokenize(a))
#
# a = nltk.pos_tag(nltk.word_tokenize(a))
# print(a)
#
#
#
# grammar = r"""
#   PP: {<IN><PRP>}               # Chunk prepositions followed by NP
#   """
# NP_grammar = "NP: {<DT>?<JJ>*<NN>}"
# PP_grammar = "PP: {<IN><PRP>}"
# IV_grammar = 'IV: {<TO><VB>}'
# NP = nltk.RegexpParser(NP_grammar)
# cp = nltk.RegexpParser(grammar)
# IV = nltk.RegexpParser(IV_grammar)
# Tree = NP.parse(a)
# print(Tree)
# parser = nltk.parse.malt.MaltParser()
# Tree = cp.parse(a)
# print(Tree)
#NP parser grammer
NP_grammar = "NP: {<DT>?<JJ>*<NN>}"
PP_grammar = "PP: {<IN><PRP>}"
IV_grammar = 'IV: {<TO><VB>}'
NP = nltk.RegexpParser(NP_grammar)
PP = nltk.RegexpParser(PP_grammar)
IV = nltk.RegexpParser(IV_grammar)
features = []
Sentence = np.array(pd.read_csv('it-corpus.tsv', sep = '\t').Sentence)
Label = np.array(pd.read_csv('it-corpus.tsv', sep = '\t').Class)
raw = Label + '|' + Sentence
raw = np.unique(raw)
Sentence = []
Label = []
for i in range(len(raw)):

    Label.append(raw[i].split('|')[0])
    Sentence.append(raw[i].split('|')[1])


for ind, i in enumerate(Sentence):
    for jnd, j in enumerate(Sentence):
        if ind != jnd and i == j:
            if Label[ind] == 'NomAnaph':
                del Label[ind]
                del Sentence[ind]
            else:
                del Label[jnd]
                del Sentence[jnd]
Label = np.array(Label)
Sentence = np.array(Sentence)

raw = Sentence
F1 = []
F2 = []
F3 = []
F4 = []
F5 = []
F6 = []
F7 = []
F7_1 = []
F7_2 = []
F7_3 = []
F7_4 = []
F7_5 = []
F7_6 = []
F7_7 = []
F7_8 = []
F8 = []
F9 = []
F10 = []
F11 = []
F12 = []
F13 = []
F14 = []
F15 = []
F16 = []
F17 = []
F18 = []
F19 = []
F20 = []
s_sentence = []
s_label = []
for ind, i in enumerate(raw):
    words = nltk.word_tokenize(i)
    for jdex, j in enumerate(words):

        if j.lower() == 'it':
            s_sentence.append(i)
            s_label.append(Label[ind])

            # feature = []

            punc_num = 0
            #F1 Position of it.
            F1.append(jdex + 1)
            #F2 Number of tokens.
            F2.append(len(words))
            #F3 Number of punctuations
            tags = nltk.pos_tag(words)
            for k in tags:
                if len(k) == 1 and k[1] in string.punctuation:
                    punc_num += 1
            F3.append(punc_num)
            #F4 NP number before it
            if jdex == 0:
                F4.append(0)
            else:
                tags = nltk.pos_tag(words[:jdex])
                NP_tree = NP.parse(tags)

                num_NP = find_NP(NP_tree)
                F4.append(num_NP)
            #F5 NP number after it
            if jdex == len(words) - 1:
                F5.append(0)
            else:
                tags = nltk.pos_tag(words[jdex+1:])
                NP_tree = NP.parse(tags)

                num_NP = find_NP(NP_tree)
                F5.append(num_NP)
            # F6 Followed by a Prepositional phrase or not !!!
            tags = nltk.pos_tag(words[:jdex])
            PP_tree = PP.parse(tags)
            F6.append(PP_follow(PP_tree))
            # F7The part-of-speech (POS) tags of the four tokens immediately preceding and the four tokens immediately succeeding a given instance of “it”
            tags = nltk.pos_tag(words)
            resl = []
            resr = []
            if jdex - 4 >= 0:
                for k in range(jdex - 4, jdex):
                    resl.append(tags[k][1])
            else:
                for k in range(0, 4 - jdex):
                    resl.append('ABS')
                for k in range(0, jdex):
                    resl.append(tags[k][1])

            if len(tags) - 1 - jdex >= 4:
                for k in range(jdex + 1, jdex + 5):
                    resr.append(tags[k][1])
            else:
                for k in range(jdex + 1, len(tags)):
                    resr.append(tags[k][1])
                for k in range(0, 4 - (len(tags) - 1 - jdex)):
                    resr.append('ABS')
            # feature = feature + resl + resr
            F7_1.append(resl[0])
            F7_2.append(resl[1])
            F7_3.append(resl[2])
            F7_4.append(resl[3])
            F7_5.append(resr[0])
            F7_6.append(resr[1])
            F7_7.append(resr[2])
            F7_8.append(resr[3])
            # F8 Is the occurrence of “it” followed by an -ing form of a verb?
            tags = nltk.pos_tag(words[jdex:])
            ans = False
            for k in tags:
                if k[1] == 'VBG':
                    ans = True
                    break
            F8.append(ans)
            # F9 Is the occurrence of “it” followed by a preposition? (Yes/No)
            tags = nltk.pos_tag(words[jdex:])
            ans = False
            for k in tags:
                if k[1] == 'IN':
                    ans = True
                    break
            F9.append(ans)
            # F10 The number of adjectives that follow the occurrence of “it” in the sentence
            tags = nltk.pos_tag(words[jdex:])
            count = 0
            for k in tags:
                if k[1] == 'JJ':
                    count += 1
            F10.append(count)
            # F11 Is the pronoun “it” preceded by a verb? (Yes/No)
            flag = False
            tags = nltk.pos_tag(words)
            for k in tags[:jdex]:
                if k[1][:2] == 'VB':
                    flag = True
                    break
            F11.append(flag)
            #F12 Is the pronoun “it” followed by a verb? (Yes/No)
            flag = False
            for k in tags[jdex:]:
                if k[1][:2] == 'VB':
                    flag = True
                    break
            F12.append(flag)
            # F13 Is the pronoun 'it' followed by an adjective.
            flag = False
            for k in tags[jdex:]:
                if k[1] == 'JJ':
                    flag = True
                    break
            F13.append(flag)
            #14 True if there is a noun phrase coming after the pronoun 'it' and that noun phrase contains an adjective otherwise false.
            tags = nltk.pos_tag(words[jdex:])
            NP_tree = NP.parse(tags)
            F14.append(NP_JJ_follow(NP_tree))
            # F15 The number of tokens coming before the following infinitive verb (if there is one), otherwise 0.
            tags = nltk.pos_tag(words)
            IV_tree = IV.parse(tags)
            F15.append(find_IV(IV_tree))
            # F16 The number of tokens that appear between the pronoun 'it' and the first following preposition
            tags = nltk.pos_tag(words[jdex:])
            F16.append(count_tokens_PP(tags))
            # F17 True if there a sequence “adjective + noun phrase” following the pronoun “it”, and false otherwise.
            tree = NP.parse(tags)
            F17.append(find_ADJ_NP(tree))
            # F18 The dependency relation type with the closest word to which “it” is associated as a dependent. !!!
            depend_parser = spacy.load("en_core_web_sm")
            sentence = re.sub("\s\s+" , " ", i)
            results = depend_parser(sentence)
            F18.append(str(results[jdex].dep_))
            # F19 True if the immediately following verb belongs to the category “weather adjectives”, and false otherwise.

            ans = False
            tags = nltk.pos_tag(words)
            if jdex + 1 < len(words) and tags[jdex + 1][1][:2] == 'VB':
                for synset in wn.synsets(tags[jdex + 1][0]):
                    if synset.lexname() == 'verb.weather':
                        ans = True
            F19.append(ans)
            # F20 True if the immediately following verb belongs to the category of cognitive verbs, and false otherwise.
            ans = False
            tags = nltk.pos_tag(words)
            if jdex + 1 < len(words) and tags[jdex + 1][1][:2] == 'VB':
                for synset in wn.synsets(tags[jdex + 1][0]):
                    if synset.lexname() == 'verb.cognition':
                        ans = True
            F20.append(ans)
            #features.append(feature)
# print(features)
# print(Sentence)

df = {}
df['Labels'] = s_label
df['Sentences'] = s_sentence
df['F1'] = F1
df['F2'] = F2
df['F3'] = F3
df['F4'] = F4
df['F5'] = F5
df['F6'] = F6
df['F7_1'] = F7_1
df['F7_2'] = F7_2
df['F7_3'] = F7_3
df['F7_4'] = F7_4
df['F7_5'] = F7_5
df['F7_6'] = F7_6
df['F7_7'] = F7_7
df['F7_8'] = F7_8
df['F8'] = F8
df['F9'] = F9
df['F10'] = F10
df['F11'] = F11
df['F12'] = F12
df['F13'] = F13
df['F14'] = F14
df['F15'] = F15
df['F16'] = F16
df['F17'] = F17
df['F18'] = F18
df['F19'] = F19
df['F20'] = F20

df = pd.DataFrame(df)
df.to_csv('results.csv')

