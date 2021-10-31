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

from spacy_wordnet.wordnet_annotator import WordnetAnnotator

# a = "I want to buy some napkins of dog. What's in the door. I am a good man"
# print(nltk.word_tokenize(a))
#
# a = nltk.pos_tag(nltk.word_tokenize(a))
# print(a)
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
#print(Tree)
#parser = nltk.parse.malt.MaltParser()
# Tree = cp.parse(a)
#print(Tree)


#NP parser grammer
NP_grammar = "NP: {<DT>?<JJ>*<NN>}"
PP_grammar = "PP: {<IN><PRP>}"
IV_grammar = 'IV: {<TO><VB>}'
NP = nltk.RegexpParser(NP_grammar)
PP = nltk.RegexpParser(PP_grammar)
IV = nltk.RegexpParser(IV_grammar)
features = []
raw = np.unique(np.array(pd.read_csv('it-corpus.tsv', sep = '\t').Sentence))
for ind, i in enumerate(raw):
    words = nltk.word_tokenize(i)
    for jdex, j in enumerate(words):

        if j.lower() == 'it':
            feature = []
            punc_num = 0
            #F1 Position of it.
            feature.append(jdex + 1)
            #F2 Number of tokens.
            feature.append(len(words))
            #F3 Number of punctuations
            tags = nltk.pos_tag(words)
            for k in tags:
                if len(k) == 1 and k[1] in string.punctuation:
                    punc_num += 1
            feature.append(punc_num)
            #F4 NP number before it
            if jdex == 0:
                feature.append(0)
            else:
                tags = nltk.pos_tag(words[:jdex])
                NP_tree = NP.parse(tags)

                num_NP = find_NP(NP_tree)
                feature.append(num_NP)
            #F5 NP number after it
            if jdex == len(words) - 1:
                feature.append(0)
            else:
                tags = nltk.pos_tag(words[jdex+1:])
                NP_tree = NP.parse(tags)

                num_NP = find_NP(NP_tree)
                feature.append(num_NP)
            # F6 Followed by a Prepositional phrase or not !!!
            tags = nltk.pos_tag(words[:jdex+1])
            PP_tree = PP.parse(tags)
            feature.append(PP_follow(PP_tree))
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
            feature = feature + resl + resr
            # F8 Is the occurrence of “it” followed by an -ing form of a verb?
            tags = nltk.pos_tag(words[jdex:])
            ans = False
            for k in tags:
                if k[1] == 'VBG':
                    ans = True
                    break
            feature.append(ans)
            # F9 Is the occurrence of “it” followed by a preposition? (Yes/No)
            tags = nltk.pos_tag(words[jdex:])
            ans = False
            for k in tags:
                if k[1] == 'IN':
                    ans = True
                    break
            feature.append(ans)
            # F10 The number of adjectives that follow the occurrence of “it” in the sentence
            tags = nltk.pos_tag(words[jdex:])
            count = 0
            for k in tags:
                if k[1] == 'JJ':
                    count += 1
            feature.append(count)
            # F11 Is the pronoun “it” preceded by a verb? (Yes/No)
            flag = False
            tags = nltk.pos_tag(words)
            for k in tags[:jdex]:
                if k[1][:2] == 'VB':
                    flag = True
                    break
            feature.append(flag)
            #F12 Is the pronoun “it” followed by a verb? (Yes/No)
            flag = False
            for k in tags[jdex:]:
                if k[1][:2] == 'VB':
                    flag = True
                    break
            feature.append(flag)
            # F13 Is the pronoun 'it' followed by an adjective.
            flag = False
            for k in tags[jdex:]:
                if k[1][:2] == 'J':
                    flag = True
                    break
            feature.append(flag)
            #14 True if there is a noun phrase coming after the pronoun 'it' and that noun phrase contains an adjective otherwise false.
            tags = nltk.pos_tag(words[jdex:])
            NP_tree = NP.parse(tags)
            feature.append(NP_JJ_follow(NP_tree))
            # F15 The number of tokens coming before the following infinitive verb (if there is one), otherwise 0.
            tags = nltk.pos_tag(words)
            IV_tree = IV.parse(tags)
            feature.append(find_IV(IV_tree))
            # F16 The number of tokens that appear between the pronoun 'it' and the first following preposition
            tags = nltk.pos_tag(words[jdex:])
            feature.append(count_tokens_PP(tags))
            # F17 True if there a sequence “adjective + noun phrase” following the pronoun “it”, and false otherwise.
            tree = NP.parse(tags)
            feature.append(find_ADJ_NP(tree))
            # F18 The dependency relation type with the closest word to which “it” is associated as a dependent. !!!
            depend_parser = spacy.load("en_core_web_sm")
            sentence = re.sub("\s\s+" , " ", i)
            results = depend_parser(sentence)
            feature.append(str(results[jdex].dep_))
            # F19 True if the immediately following verb belongs to the category “weather adjectives”, and false otherwise.

            ans = False
            tags = nltk.pos_tag(words)
            if jdex + 1 < len(words) and tags[jdex + 1][1][:2] == 'VB':
                for synset in wn.synsets(tags[jdex + 1][0]):
                    if synset.lexname() == 'verb.weather':
                        ans = True
            feature.append(ans)
            # F20 True if the immediately following verb belongs to the category of cognitive verbs, and false otherwise.
            ans = False
            tags = nltk.pos_tag(words)
            if jdex + 1 < len(words) and tags[jdex + 1][1][:2] == 'VB':
                for synset in wn.synsets(tags[jdex + 1][0]):
                    if synset.lexname() == 'verb.cognition':
                        ans = True
            feature.append(ans)
            features.append(feature)
print(features)
