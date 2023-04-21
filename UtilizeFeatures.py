from itertools import product
from re import finditer
from nltk.tokenize import RegexpTokenizer
import re
import math
import ngram
from fuzzycomp.fuzzycomp import fuzzycomp
from gensim.models import KeyedVectors
from nltk.corpus import wordnet
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from tqdm import tqdm
import pandas as pd
import yaml
import random
from CreateTrainingSet import get_dataset
import os


print('Loading word2vec model...')
model = KeyedVectors.load_word2vec_format('D:\Docs\programs\TrentoLab\OM/GoogleNews-vectors-negative300.bin',
                                          binary=True)
print('Word2vec model are loaded.')

def readFiles(tpath):
    txtLists = os.listdir(tpath)
    List = []
    for txt in txtLists:
        if txt[0:2] != "._":
            List.append(tpath + txt)

    return List


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
                       identifier)
    return [m.group(0) for m in matches]


def get_word2vec_sim(row_set1, row_set2):
    sum_sim = 0
    N = max(len(row_set1), len(row_set2))

    for w1 in row_set1:
        maxSim = 0
        for w2 in row_set2:
            try:
                sim = model.wv.similarity(w1, w2)
            except:
                sim = 0

            if sim > maxSim:
                maxSim = sim
        if maxSim > sum_sim:
            sum_sim = maxSim

    return sum_sim


def get_words(text):
    if '_' in text:
        row_set = text.split('_')
    else:
        if '-' in text:
            row_set = text.split('-')
        else:
            row_set = camel_case_split(text)

    row_set = [x.lower() for x in row_set]
    return row_set


def StringSimilarities(dataset):
    ngrams1 = []
    lcs = []
    lws = []
    wordnet_sims = []
    w2vec_sims = []

    index = 2

    for key, row in tqdm(dataset.iterrows()):

        string1 = row[index]
        string2 = row[index + 1]
        # print(string1, string2)

        ngrams1.append(ngram.NGram.compare(string1, string2, N=1))
        lws.append(lev.get_sim_score(string1, string2))
        lcs.append(2 * fuzzycomp.lcs_length(string1, string2) / (len(string1) + len(string2)))
        row_set1 = get_words(string1)
        row_set2 = get_words(string2)

        allsyns1 = set(ss for word in row_set1 for ss in wordnet.synsets(word))
        allsyns2 = set(ss for word in row_set2 for ss in wordnet.synsets(word))

        best = [wordnet.wup_similarity(s1, s2) for s1, s2 in
                product(allsyns1, allsyns2)]
        best = list(filter(None, best))
        if len(best) > 0:
            best.sort(reverse=True)
            wordnet_sims.append(best[0])
        else:
            wordnet_sims.append(0)

        w2vec_sims.append(get_word2vec_sim(row_set1, row_set2))

    dataset['Ngram1'] = ngrams1
    dataset['Longest_com_sub'] = lcs
    dataset['Levenshtein'] = lws
    dataset['Wordnet_sim'] = wordnet_sims
    dataset['Word2vec_sim'] = w2vec_sims

    return dataset


def formal_property(prop):
    x = str(prop)
    # Fliter Dashes
    # x =x.replace('-','')

    # Fliter upper cases
    r = re.compile('[A-Z]*[a-z]*[_]*\d*')
    x = " ".join(r.findall(x))
    x = x.replace("_", "")

    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(x)
    tokens = [token.lower() for token in tokens]
    # lemmatiser = WordNetLemmatizer()
    # lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]
    x = "".join(tokens)
    x = x.replace("_", "")
    x = x.replace(" ", "")

    return x


def minmax_norm(data):
    Min = min(data)
    Max = max(data)
    if Min == Max:
        return [random.random() for _ in range(len(data))]
    for i in range(len(data)):
        data[i] = (data[i] - Min) / (Max - Min)
    return data


def Z_Score(data):
    lenth = len(data)
    total = sum(data)
    ave = float(total) / lenth
    tempsum = sum([pow(data[i] - ave, 2) for i in range(lenth)])
    tempsum = pow(float(tempsum) / lenth, 0.5)
    if tempsum == 0:
        return [random.random() for _ in range(len(data))]
    for i in range(lenth):
        data[i] = (data[i] - ave) / tempsum
        if data[i] > 1: data[i] = 1
        if data[i] < 0: data[i] = 0
    return data


def log_norm(data):
    Max = max(data)
    data = [math.log(10, x) / math.log(10, Max) for x in data]
    return data


def sigmoid_norm(data):
    for i in range(len(data)):
        data[i] = 1.0 / (1 + math.exp(-data[i]))
    return data

def string_based_matching(row):
    # strategy for align properties, by string similarity
    F = False
    if formal_property(row['Label1']) == formal_property(row['Label2']):
        F = True
    # if row['Ngram1'] > 0.9:
    #     F = True
    # elif row['Longest_com_sub'] > 0.9:
    #     F = True
    # elif row['Levenshtein'] > 0.9:
    #     F = True
    # elif row['Wordnet_sim'] > 0.95:
    #     F = True
    # elif row['Word2vec_sim'] > 0.95:
    #     F = True
    # elif PropertyPair_ML(row):
    # # Machine learning based property alignment prediction
    #     F = True
    return F



def present(pairs, PropPair_tag):
    pp = dict()
    prop_pairs = pairs.loc[(pairs['Type'] == "Property")]
    if PropPair_tag:
        # predict aligned properties in a blind scenario
        prop_pairs = StringSimilarities(prop_pairs)
        for index, row in prop_pairs.iterrows():
            if string_based_matching(row):
                pp[row['Label1']] = row['Label2']
    else:
        for index, row in prop_pairs.loc[(prop_pairs['Match'] == 1)].iterrows():
            pp[row['Label1']] = row['Label2']

    return pp


def addVS(Etypes, name1, name2, propertyPairs, negative_tag=True):
    FCA1 = pd.read_csv("FCA/" + DATASET + "/%s_FCA-v.csv" % name1)
    FCA2 = pd.read_csv("FCA/" + DATASET + "/%s_FCA-v.csv" % name2)

    ES = []
    for key, row in tqdm(Etypes.iterrows()):
        string1 = row[2]
        string2 = row[3]

        if string1 in FCA1['EType'].values and string2 in FCA2['EType'].values:
            score = 0

            for p1, p2 in propertyPairs.items():
                if p1 in FCA1.columns.values.tolist() and p2 in FCA2.columns.values.tolist():
                    s1 = FCA1[FCA1['EType'] == string1][p1].values[0]
                    s2 = FCA2[FCA2['EType'] == string2][p2].values[0]

                    if negative_tag:
                        if s1 > 0 and s2 > 0:
                            score += (s1 + s2) / 2
                        elif s1 < 0 < s2:
                            score += (-1 - s2) / 2
                        elif s2 < 0 < s1:
                            score += (-1 - s2) / 2
                    else:
                        score += (s1 + s2) / 2

            if score <= 0: score = 0
            ES.append(math.log(1 + 0.2 * score))

        else:
            ES.append(0)

    if len(ES) > 0:
        ES = minmax_norm(ES)

    return ES


def addHS(Etypes, name1, name2, propertyPairs, negative_tag=True):
    FCA1 = pd.read_csv("FCA/" + DATASET + "/%s_FCA-h.csv" % name1)
    FCA2 = pd.read_csv("FCA/" + DATASET + "/%s_FCA-h.csv" % name2)

    ES = []
    for key, row in tqdm(Etypes.iterrows()):
        string1 = row[2]
        string2 = row[3]

        if string1 in FCA1['EType'].values and string2 in FCA2['EType'].values:
            score = 0

            for p1, p2 in propertyPairs.items():
                if p1 in FCA1.columns.values.tolist() and p2 in FCA2.columns.values.tolist():
                    s1 = FCA1[FCA1['EType'] == string1][p1].values[0]
                    s2 = FCA2[FCA2['EType'] == string2][p2].values[0]
                    nproperty1 = FCA1[FCA1['EType'] == string1]['NProperty'].values[0]
                    nproperty2 = FCA2[FCA2['EType'] == string2]['NProperty'].values[0]

                    if negative_tag:
                        if s1 >= 0 and s2 >= 0:
                            w1 = math.exp(0.2 * (1 - s1))
                            w2 = math.exp(0.2 * (1 - s2))
                            score += (w1 / nproperty1 + w2 / nproperty2) / 2
                        elif s1 < 0 < s2:
                            w1 = -1
                            w2 = math.exp(0.2 * (1 - s2))
                            score += (w1 / nproperty1 - w2 / nproperty2) / 2
                        elif s2 < 0 < s1:
                            w1 = math.exp(0.2 * (1 - s1))
                            w2 = -1
                            score += (-w1 / nproperty1 + w2 / nproperty2) / 2
                    else:
                        w1 = math.exp(0.2 * (1 - s1))
                        w2 = math.exp(0.2 * (1 - s2))
                        score += (w1 / nproperty1 + w2 / nproperty2) / 2

            # if score >= 1:
            #     score = 1
            ES.append(score)

        else:
            ES.append(0)

    if len(ES) > 0:
        ES = Z_Score(ES)

    return ES


def addIS(Etypes, name1, name2, propertyPairs, negative_tag=True):
    FCA1 = pd.read_csv("FCA/" + DATASET + "/%s_FCA-i.csv" % name1)
    FCA2 = pd.read_csv("FCA/" + DATASET + "/%s_FCA-i.csv" % name2)

    ES = []
    for key, row in tqdm(Etypes.iterrows()):
        string1 = row[2]
        string2 = row[3]

        if string1 in FCA1['EType'].values and string2 in FCA2['EType'].values:
            score = 0

            for p1, p2 in propertyPairs.items():
                if p1 in FCA1.columns.values.tolist() and p2 in FCA2.columns.values.tolist():
                    s1 = FCA1[FCA1['EType'] == string1][p1].values[0]
                    s2 = FCA2[FCA2['EType'] == string2][p2].values[0]
                    nproperty1 = FCA1[FCA1['EType'] == string1]['NProperty'].values[0]
                    nproperty2 = FCA2[FCA2['EType'] == string2]['NProperty'].values[0]

                    if negative_tag:
                        if s1 >= 0 and s2 >= 0:
                            w1 = s1
                            w2 = s2
                            score += (w1 / nproperty1 + w2 / nproperty2) / 2
                        elif s1 < 0 < s2:
                            w1 = -1
                            w2 = s2
                            score += (w1 / nproperty1 - w2 / nproperty2) / 2
                        elif s2 < 0 < s1:
                            w1 = s1
                            w2 = -1
                            score += (-w1 / nproperty1 + w2 / nproperty2) / 2
                    else:
                        score += (s1 / nproperty1 + s2 / nproperty2) / 2

            # if score >= 1:
            #     score = 1
            ES.append(score)

        else:
            ES.append(0)

    if len(ES) > 0:
        ES = Z_Score(ES)

    return ES


def generate_datasetsForML(alignment, datasetname, PropPair_tag):
    KG1 = alignment.split('/')[-1].split('.')[0].split('-')[0]
    KG2 = alignment.split('/')[-1].split('.')[0].split('-')[1]
    pairs = get_dataset(alignment, datasetname, PropPair_tag, m_factor=10)

    # Build Etype-Etype or Etype-entity pairs
    Etypes = pairs[pairs['Type'] == "Etype"]
    # Generate String Similarities between these candidate pairs
    Etypes = StringSimilarities(Etypes)

    # Collect paired properties (from GT dataset or by ETR itself)
    propertyPairs = present(pairs, PropPair_tag)

    # Generate our property-based Similarities between these candidate pairs
    Etypes['Sim_V'] = addVS(Etypes, KG1, KG2, propertyPairs)
    Etypes['Sim_H'] = addHS(Etypes, KG1, KG2, propertyPairs)
    Etypes['Sim_I'] = addIS(Etypes, KG1, KG2, propertyPairs)

    return Etypes

"""
To produce the evaluation dataset for training and testing the ML model, we need to know the GT alignments of Etype-Etype, or Etype-Entity.
Assume we have two scenario: controlled by PropPair_tag
1. We only know the target alignments, i.e.  GT alignments of Etype-Etype, or Etype-Entity. The property alignments need to be recognized by ourselves. 
2. We also know the property alignments, that is we have GT alignments of Etype-Etype/Etype-Entity and property-property
"""

# Generate features for training dataset
DATASET = 'Conference'
KG_Alignments = readFiles('D:\Docs\programs\TrentoLab\OM\Data/%s/reference/' % DATASET)
PropPair_tag = False
lev = Levenshtein()

flag = 0
for alignment in KG_Alignments:
    Etypes = generate_datasetsForML(alignment, DATASET, PropPair_tag)
    add = "D:\Docs\programs\TrentoLab\OM\Entity type recognition/TrainingSets/" + DATASET + '_features.csv'
    if flag == 0:
        Etypes.to_csv(add, index=False)
        flag += 1
    else:
        Etypes.to_csv(add, index=False, mode='a', header=False)
