import itertools
import yaml
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from owlready2 import get_ontology
from sklearn.metrics import f1_score
import random
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

lev = Levenshtein()

# It's long
print('Loading word2vec model...')
model = KeyedVectors.load_word2vec_format('G:\BACKUP\learning\programs\LiveSchema\OM/GoogleNews-vectors-negative300.bin',binary=True)
print('Word2vec model are loaded.')

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
        data[i] = (data[i]-Min)/(Max-Min)
    return data

def Z_Score(data):
    lenth = len(data)
    total = sum(data)
    ave = float(total)/lenth
    tempsum = sum([pow(data[i] - ave,2) for i in range(lenth)])
    tempsum = pow(float(tempsum)/lenth,0.5)
    if tempsum == 0:
        return [random.random() for _ in range(len(data))]
    for i in range(lenth):
        data[i] = (data[i] - ave)/tempsum
        if data[i] > 1: data[i] = 1
        if data[i] < 0: data[i] = 0
    return data

def log_norm(data):
    Max = max(data)
    data = [math.log(10,x)/math.log(10,Max) for x in data]
    return data

def sigmoid_norm(data):
    for i in range(len(data)):
        data[i] = 1.0/(1+math.exp(-data[i]))
    return data

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

def read_class(class_path):
    classes = []
    EtypeNames = pd.read_csv(class_path).head(500)["EtypeName"]
    for e in EtypeNames:
        classes.append(e)
    return classes


def read_property(property_path):
    properties = []
    PropertyNames = set(pd.read_csv(property_path)["PropertyName"])
    for e in PropertyNames:
        properties.append(e)
    return properties


def get_mappings(filename, Inverse_flag = 0):
    mappings = []

    with open(filename) as f:
        soup = BeautifulSoup(f, 'xml')

    cells = soup.find_all('Cell')

    for cell in cells:
        entity1 = cell.find('entity1').attrs['rdf:resource'].split('#')[1]
        entity2 = cell.find('entity2').attrs['rdf:resource'].split('#')[1]
        if Inverse_flag == 0:
            mappings.append((entity1, entity2))
        else:
            mappings.append((entity2, entity1))

    return mappings


def get_dataset():
    data = []
    data_unmatched = []
    ont1_path_c = 'Input/general/'  + 'lov_dbpedia-owl_class.csv'
    ont2_path_c = 'Input/general/'  + 'github_sumo_human.csv'
    ont1_path_p = 'Input/general/'  + 'lov_dbpedia-owl_property.csv'
    ont2_path_p = 'Input/general/'  + 'github_sumo_property.csv'

    # Parse ontologies
    classes1 = read_class(ont1_path_c)
    properties1 = read_property(ont1_path_p)
    classes2 = read_class(ont2_path_c)
    properties2 = read_property(ont2_path_p)

    target_cls = ["Person"]
    class_pairs_p = list(itertools.product(target_cls, classes2))
    # print(class_pairs_p)
    for class_pair in class_pairs_p:
        pair = (class_pair[0], class_pair[1])
        match = 1
        data.append((ont1_path_c.split(".")[0].split("_")[0], ont2_path_c.split(".")[0].split("_")[0], pair[0],
                         pair[1], match, 'Etype'))

    class_pairs_n = list(itertools.product(classes1, classes2))
    for class_pair in class_pairs_n:
        pair = (class_pair[0], class_pair[1])
        match = 0
        data_unmatched.append(
            (ont1_path_c.split(".")[0].split("_")[0], ont2_path_c.split(".")[0].split("_")[0], pair[0],
                 pair[1], match, 'Etype'))

    data = data + random.sample(data_unmatched, len(data))
    random.shuffle(data)

    properties_pairs = list(itertools.product(properties1, properties2))

    for prop_pair in properties_pairs:
        pair = (prop_pair[0], prop_pair[1])
        string_sim = lev.get_sim_score(prop_pair[0], prop_pair[1])

        row_set1 = get_words(prop_pair[0])
        row_set2 = get_words(prop_pair[1])

        word_sim = get_word2vec_sim(row_set1, row_set2)
        if string_sim == 1:
            match = 1
            data.append(
                (ont1_path_p.split(".")[0].split("_")[0], ont2_path_p.split(".")[0].split("_")[0], pair[0], pair[1],
                 match, 'Property'))
            print(pair)


    dataset = pd.DataFrame(data, columns=['Ontology1', 'Ontology2', 'Name1',
                                          'Name2','Match', 'Type'])

    return dataset

def present(Etypes, property):
    propertyPairs = property[property['Match'] == 1]
    pp = dict()
    for index, row in propertyPairs.iterrows():
        pp[formal_property(row[2])] = formal_property(row[3])
    print(pp)

    return pp

def addVS(Etypes, data):
    FCA1 = pd.read_csv("FCA2/general/lov_dbpedia-owl_FCA-v.csv")
    Properties = data[data['Type'] == "Property"]

    # for presenting the Etype pairs and properties pairs
    propertyPairs = present(Etypes, Properties)

    ES = []
    for key, row in tqdm(Etypes.iterrows()):
        string1 = formal_property(row[2])

        if string1 in FCA1['EType'].values :
            score = 0

            for p1, p2 in propertyPairs.items():
                if p1 in FCA1.columns.values.tolist():
                    s1 = FCA1[FCA1['EType'] == string1][p1].values[0]
                    if s1 > 0 :
                        score += (s1) / 2
                    elif s1 < 0 :
                        score += (0) / 2


            if score <= 0: score = 0
            ES.append(math.log(1 + 0.2 * score))

        else:
            ES.append(0)

    if len(ES)>0:
        ES = minmax_norm(ES)

    return ES


def addHS(Etypes, data):
    FCA1 = pd.read_csv("FCA2/general/lov_dbpedia-owl_FCA-h.csv")
    Properties = data[data['Type'] == "Property"]

    # for presenting the Etype pairs and properties pairs
    propertyPairs = present(Etypes, Properties)

    ES = []
    for key, row in tqdm(Etypes.iterrows()):
        string1 = formal_property(row[2])

        if string1 in FCA1['EType'].values:
            score = 0

            for p1, p2 in propertyPairs.items():
                if p1 in FCA1.columns.values.tolist() :
                    s1 = FCA1[FCA1['EType'] == string1][p1].values[0]
                    nproperty1 = FCA1[FCA1['EType'] == string1]['NProperty'].values[0]

                    if s1 >= 0 :
                        w1 = math.exp(0.2 * (1 - s1))
                        score += (w1 / nproperty1) / 2
                    elif s1 < 0 :
                        w1 = 0
                        score += (w1 / nproperty1 ) / 2


            if score >= 1:
                score = 1

            ES.append(score)

        else:
            ES.append(0)

    if len(ES) > 0:
        ES = Z_Score(ES)

    return ES

def addIS(Etypes, data):
    FCA1 = pd.read_csv("FCA2/general/lov_dbpedia-owl_FCA-i.csv")
    Properties = data[data['Type'] == "Property"]

    # for presenting the Etype pairs and properties pairs
    propertyPairs = present(Etypes, Properties)

    ES = []
    for key, row in tqdm(Etypes.iterrows()):
        string1 = formal_property(row[2])
        string2 = formal_property(row[3])

        if string1 in FCA1['EType'].values:
            score = 0

            for p1, p2 in propertyPairs.items():
                if p1 in FCA1.columns.values.tolist():
                    s1 = FCA1[FCA1['EType'] == string1][p1].values[0]
                    nproperty1 = FCA1[FCA1['EType'] == string1]['NProperty'].values[0]

                    if s1 >= 0 :
                        w1 = s1
                        score += (w1 / nproperty1 ) / 2
                    elif s1 < 0 :
                        w1 = 0
                        score += (w1 / nproperty1) / 2


            if score >= 1:
                score = 1
            ES.append(score)

        else:
            ES.append(0)

    if len(ES) > 0:
        ES = Z_Score(ES)

    return ES

def generate_datasetsForML():
    data = get_dataset()
    # data = pd.read_csv("data/" + SELECTED_DATASET + '_' + name1 + '_' + name2 + '.csv')
    Etypes = data[data['Type'] == "Etype"]

    Etypes['Sim_V'] = addVS(Etypes,  data)
    Etypes['Sim_H'] = addHS(Etypes,  data)
    Etypes['Sim_I'] = addIS(Etypes,  data)
    return Etypes

Etypes = generate_datasetsForML()
Etypes.to_csv("dataWithFeatures/general_test_features.csv", index=False)


