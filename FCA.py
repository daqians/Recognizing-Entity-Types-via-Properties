###TODO list
# 1. distinguish property terms separated or unified (DONE)
# 2. distinguish the reading process of ET_P mapping for schema and entity (notice the object property and data property) (DONE)
# 3. Need to consider the properties pair generation when doing blind ETR in a instance-level case (DONE)
# 4. The reading order of the input alignments like cmt-conference and conference-cmt (DONE)


from scipy.stats import entropy
from math import log, e
import pandas as pd
from pandas import Series
import numpy as np
import re
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('wordnet')


def GetMapping_E_P(triples):
    # Create the DataFrame to save the vocabs' triples with the predicates present on the argument 'predicates'
    df = pd.DataFrame(columns=["EType", "Property", "FCATerms"])

    # Get the list of predicates
    strPredicates = ["domain", "domainIncludes", "range"]

    # Create the list used to add the triples that has that predicate
    list_ = list()
    EtypeList = list()
    index_ = 0
    # Iterate for every triples present on the file passed on the argument 'triples'
    for index, row in triples.iterrows():
        # if a triple has a specified PredicateTerm or the predicates are not set
        if str(row["PredicateTerm"]) in strPredicates:
            # Save that triple on the list

            if "|" in str(row["ObjectTerm"]):
                for obs in str(row["ObjectTerm"]).split("|"):
                    t = obs
                    x = row["SubjectTerm"]
                    list_.insert(index_, {"EType": t, "Property": row["SubjectTerm"], "FCATerms": x})
                    index_ += 1
            else:
                t = row["ObjectTerm"]
                x = row["SubjectTerm"]
                list_.insert(index_, {"EType": t, "Property": row["SubjectTerm"], "FCATerms": x})
                index_ += 1

        if str(row["PredicateTerm"]) == "type" and str(row["ObjectTerm"]) == "Class" and "http" in str(row["Subject"]):
            EtypeList.append(row["SubjectTerm"])

    if (index_ and len(list_)):
        df = df.append(list_)

    return df, EtypeList

def GetMapping_ET_P(triples):
    # Create the DataFrame to save the vocabs' triples with the predicates present on the argument 'predicates'
    EtypePropertyList = pd.DataFrame(columns=["EType", "Property", "FCATerms"])

    # Get the list of predicates
    strPredicates = ["domain", "domainIncludes", "range"]

    # Create the list used to add the triples that has that predicate
    list_ = list()
    EtypeList = list()
    index_ = 0

    # Iterate for every triples present on the file passed on the argument 'triples'
    for index, row in triples.iterrows():
        # if a triple has a specified PredicateTerm or the predicates are not set
        if (str(row["PredicateTerm"]) in strPredicates):
            if "|" in str(row["ObjectTerm"]):
                for obs in str(row["ObjectTerm"]).split("|"):
                    t = obs
                    # distinguish property terms separated or unified
                    x = row["SubjectTerm"]
                    list_.insert(index_, {"EType": t, "Property": row["SubjectTerm"], "FCATerms": x})
                    index_ += 1
            else:
                t = row["ObjectTerm"]
                x = row["SubjectTerm"]
                list_.insert(index_, {"EType": t, "Property": row["SubjectTerm"], "FCATerms": x})
                index_ += 1

        if str(row["PredicateTerm"]) == "type" and str(row["ObjectTerm"]) == "Class" and "http" in str(row["Subject"]):
            EtypeList.append(row["SubjectTerm"])

    if (index_ and len(list_)):
        EtypePropertyList = EtypePropertyList.append(list_)

    return EtypePropertyList, EtypeList

def GetSchemaHierarchy(triples, EtypePropertyList):
    Hierarchy = pd.DataFrame(columns=["EType", "ETypeName", "SuperClass","Sub-Class", "Path", "AllSubClasses", "Layer", "Type"])
    # Properties = pd.DataFrame(columns=["Property", "PropertyName", "Domain","Range", "Type"])
    list_h = list()

    SonFatherList = dict()
    FatherSonList = dict()

    for index, row in triples.iterrows():
        if ("subclass" in str(row["PredicateTerm"]).lower()):
            if "|" not in row['ObjectTerm']:
                t = row["ObjectTerm"]
                x = row["SubjectTerm"]
                try:
                    SonFatherList[x].append(t)
                except:
                    SonFatherList[x] = [t]
                try:
                    FatherSonList[t].append(x)
                except:
                    FatherSonList[t] = [x]


    _SonFatherList = AddInheritance_fatherList(SonFatherList)
    _FatherSonList = AddInheritance_sonList(FatherSonList)
    layers = getLayers(EtypePropertyList, SonFatherList, FatherSonList)

    AllEtypes = set.union(set(EtypePropertyList["EType"]), set(_SonFatherList.keys()), set(_FatherSonList.keys()))
    index_ = 0
    for Etype in AllEtypes:
        # Notice that the _SonFatherList and _FatherSonList are not in order/path, they are randomly listed
        list_h.insert(index_, {"EType": Etype, "ETypeName": Etype, "SuperClass": SonFatherList.get(Etype, -1), "Sub-Class": FatherSonList.get(Etype, -1),
                               "Path": _SonFatherList.get(Etype, -1), "AllSubClasses": _FatherSonList.get(Etype, -1), "Layer": layers.get(Etype, 1), "Type": "class"})
        index_ += 1

    if (index_ and len(list_h)):
        Hierarchy = Hierarchy.append(list_h)

    return _SonFatherList, _FatherSonList, layers, Hierarchy

def getLayers(df,SonFatherList,FatherSonList):
    layers = dict()
    FatherSonListaa = FatherSonList.copy()
    SonFatherListaa = SonFatherList.copy()

    for index, row in df.iterrows():
        if not SonFatherListaa.get(row['EType']):
            layers[row['EType']] = 1

    queue = []
    for Etype, l in layers.items():
        queue.append(Etype)

    for q in queue:
        if FatherSonListaa.get(q):
            for son in FatherSonListaa.get(q):
                if son !=q:
                    queue.append(son)
                    layers[son] = layers[q] + 1
            FatherSonListaa.pop(q)

    return layers

# Add the subclass relations into a list

def AddInheritance_fatherList(subClassList):
    FatherList = dict()
    for son, fathers in subClassList.items():
        L = []
        findFathers(subClassList, son, L)
        FatherList[son] = L
    return FatherList

def findFathers(subClassList, son, L):
    fathers = subClassList.get(son, -1)
    if fathers == -1:
        return
    else:
        for father in fathers:
            if father == son:
                continue
            if father in L:
                continue
            L.append(father)
            findFathers(subClassList, father, L)
            # continue

        return

def AddInheritance_sonList(subClassList):
    SonList = dict()
    for father, sons in subClassList.items():
        L = []
        findSons(subClassList, father,L)
        if len(L) == 0:
            continue
        SonList[father] = L
    return SonList

def findSons(subClassList, father,L):
    sons = subClassList.get(father, -1)

    if sons == -1:
        return
    else:
        for son in sons:
            if son == father:
                continue
            if son in L:
                continue
            L.append(son)
            findSons(subClassList, son,L)
            # continue

        return


def formalize_words(Etype):
    x = str(Etype)
    # Fliter upper cases
    r = re.compile('[A-Z]*[a-z]*[_]*\d*')
    x = " ".join(r.findall(x))
    x = x.replace("_", "")

    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(x)
    tokens = [token.lower() for token in tokens]
    # lemmatiser = WordNetLemmatizer()
    # lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]
    x = " ".join(tokens)
    t = x.replace("_", "")
    t = t.replace(" ", "")

    return t


def GetCorpus(dataframe, FatherSonList, SonFatherList,_l, sep=False):
    # List all properties for each Etype/entity
    PropertyList = dict()
    for i in range(len(dataframe)):
        if dataframe["EType"][i] in PropertyList:
            PropertyList[dataframe["EType"][i]] += ' ' + dataframe["FCATerms"][i]
        elif dataframe["EType"][i] not in PropertyList:
            PropertyList[dataframe["EType"][i]] = dataframe["FCATerms"][i]

    # Words Pre-processing, set the separation strategy for the properties by sep
    for key, value in PropertyList.items():
        text = value.split()
        words = []
        for t in text:

            if sep == True:
                # t = t.replace("_", "")
                # t = t.replace("-", "")
                # t = t.strip()
                sepwords= re.findall('.[^A-Z]*', t)
                for sepw in sepwords:
                    sepw = sepw.replace("-", "")
                    words.append(sepw)
            else:
                words.append(t)

        filtered_words = [word for word in words if word.lower() not in stopwords.words('english') and not word.isdigit()]
        if len(filtered_words)>0:
            filtered_words = set(filtered_words)
            PropertyList[key] = " ".join(filtered_words)

    Corpus = [pros for _, pros in PropertyList.items() if len(pros) > 0]
    Etypes = [pros for pros, _ in PropertyList.items() if len(_) > 0]
    vectorizer = CountVectorizer(lowercase=False, token_pattern = r"[A-Za-z0-9_-]+")
    X = vectorizer.fit_transform(Corpus)
    x = X.toarray().T
    names = vectorizer.get_feature_names()
    # print(Corpus)
    # print(names)

    # Insert all info and property mappings to the dataframe
    df = pd.DataFrame()
    df["EType"] = Etypes
    df["Property"] = Corpus

    SuperClass = []
    SubClass = []
    Layers = []

    for index, row in df.iterrows():
        if FatherSonList.get(row['EType']):
            SubClass.append(FatherSonList.get(row['EType']))
        else:
            SubClass.append(set())
        if SonFatherList.get(row['EType']):
            SuperClass.append(SonFatherList.get(row['EType']))
        else:
            SuperClass.append(set())
        if _l.get(row['EType']):
            Layers.append(_l.get(row['EType']))
        else:
            Layers.append(1)

    df["SuperClass"] = SuperClass
    df["SubClass"] = SubClass
    df["Layer"] = Layers


    NCorpus = []
    for i in Corpus:
        i = str(i).split(' ')
        NCorpus.append(len(i))
    df["NProperty"] = NCorpus

    for k in range(len(x)):
        df[names[k]] = x[k]

    return df

def GetCorpus_inherited_large(dataframe):
    # dataframe = dataframe.sort_values(by='Layer')
    New = dataframe.copy(deep=True).reset_index(drop=True)
    # for index, row in New.iterrows():
    #     New.iloc[index, 6:] = New.iloc[index, 6:] * New.iloc[index, 4] + 1

    for index, column in New.iloc[:, 6:].iteritems():
        column = column * New.iloc[:, 4]
        column[column == 0] = -1
        New[index] = column

    return New

def GetCorpus_inherited(dataframe, SonFatherList,FatherSonList, EtypeList, weight_tag = True):
    dataframe = dataframe.sort_values(by='Layer')
    New = dataframe.copy(deep=True).reset_index(drop=True)

    # add positive property into FCA, W_E(P)>0
    for index, row in New.iterrows():
        New.iloc[index, 6:] = New.iloc[index, 6:] * New.iloc[index, 4]
        if SonFatherList.get(row['EType']):
            for f in SonFatherList.get(row['EType']):
                if len(dataframe[dataframe['EType'] == f]) > 0:
                    row['Property'] = row['Property'] + ' ' + dataframe[dataframe['EType'] == f]['Property'].iloc[0]
                    New.loc[index,'Property'] = row['Property']
                    kk = dataframe[dataframe['EType'] == f].iloc[0][6:] * dataframe[dataframe['EType'] == f]["Layer"].iloc[0]
                    kk[kk<=0] = 9999
                    New.iloc[index, 6:] = np.minimum(New.iloc[index,6:],kk)

        properties = set(row['Property'].split(' '))
        New.loc[index,'NProperty'] = int(len(properties))
        New.loc[index,'Property'] = str(" ".join(str(i) for i in properties))

    # add IDK into FCA
    for index, row in New.iterrows():
        if FatherSonList.get(row['EType']):
            for f in FatherSonList.get(row['EType']):
                if len(dataframe[dataframe['EType'] == f]) > 0:
                    remain = New.iloc[index,6:]

                    kk = dataframe[dataframe['EType'] == f].iloc[0][6:] * dataframe[dataframe['EType'] == f]["Layer"].iloc[0]
                    for index_column in range(6,New.shape[1]):
                        if remain[index_column-6] > 0:
                            New.iloc[index, index_column] = remain[index_column-6]
                        else:
                            New.iloc[index, index_column] = New.iloc[index, index_column] - kk[index_column - 6]

    # add negative property into FCA, IDK=0, negative property = W_E(P) * positive property
    for index, column in New.iloc[:,6:].iteritems():

        tem = int(column[column > 0].mean())
        column[column == 0] = 99999
        column[column < 0] = 0
        if weight_tag:
            column[column == 99999] = -1 * tem
        else:
            column[column == 99999] = -1
        New[index] = column

    Es = list(New['EType'])
    for E in EtypeList:
        if E not in Es:
            New = New.append({'EType': E, 'Property': '', 'SuperClass': '',
                              'SubClass': '', 'Layer': 1, 'NProperty': 0}, ignore_index=True)
            # New.loc[(New['EType'] == E)] = New.loc[(New['EType'] == E)].fillna(0)


    return New.fillna(0)

def GetFrequency(dataframe, weight_tag):
    New = dataframe.copy(deep=True)
    for index, column in New.iloc[:, 6:].iteritems():
        column[column > 0] = 1
        number = sum(column[column > 0])
        column[column > 0] = number

        tem = int(column[column > 0].mean())
        if weight_tag:
            column[column < 0] = -1 * tem
        else:
            column[column < 0] = -1

        New[index] = column
    return New


def GetEntropy(dataframe, weight_tag):
    New = dataframe.copy(deep=True)
    feature_info_gain = dict()

    for index, column in New.iloc[:, 6:].iteritems():
        column[column > 0] = 1
        New[index] = column
        column[column < 0] = 0
        New[index] = column

    for feature in New.iloc[:, 6:].columns:
        feature_info_gain[feature] = comp_feature_information_gain(New, 'EType', feature, "entropy")

    New2 = dataframe.copy(deep=True)
    for index, column in New2.iloc[:, 6:].iteritems():
        column[column > 0] = 1
        New2[index] = column * feature_info_gain[str(index)]


    return New2

def comp_feature_information_gain(df, target, descriptive_feature, split_criterion):
    """
    This function calculates information gain for splitting on
    a particular descriptive feature for a given dataset
    and a given impurity criteria.
    Supported split criterion: 'entropy', 'gini'
    """

    target_entropy = compute_impurity(df[target], split_criterion)

    entropy_list = list()
    weight_list = list()

    for level in df[descriptive_feature].unique():
        df_feature_level = df[df[descriptive_feature] == level]
        entropy_level = compute_impurity(df_feature_level[target], split_criterion)
        entropy_list.append(round(entropy_level, 3))
        weight_level = len(df_feature_level) / len(df)
        weight_list.append(round(weight_level, 3))

    feature_remaining_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))
    information_gain = target_entropy - feature_remaining_impurity

    return (information_gain)

def compute_impurity(feature, impurity_criterion):
    """
    This function calculates impurity of a feature.
    Supported impurity criteria: 'entropy', 'gini'
    input: feature (this needs to be a Pandas series)
    output: feature impurity
    """
    probs = feature.value_counts(normalize=True)

    if impurity_criterion == 'entropy':
        impurity = -1 * np.sum(np.log2(probs) * probs)
    elif impurity_criterion == 'gini':
        impurity = 1 - np.sum(np.square(probs))
    else:
        raise ValueError('Unknown impurity criterion')

    return (round(impurity, 3))

def readFiles(tpath):
    txtLists = os.listdir(tpath)

    return txtLists

def GenerateFCAs(path, tag = "schema"):
    """
    tag can be schema or instance
    if tag = schema, FCA generator process the mappings and hierarchy from the scratch, according to the the shcmea hierarchy.
    if tag = instance, FCA generator process the mappings between entities and properties, and hierarchy will inherit from its schema.
    """
    for i in readFiles(path):
        topic = path.split("/")[-1]
        name = i[:-5]
        add = path + "/" + i
        if "~" not in add and "._" not in add:
            triples = pd.read_excel(add, engine='openpyxl')
            print("process:", name)

            # if tag == "instance":
            #     hierarchyadd = path + "/" + i
            #     SonFatherList, FatherSonList, layers = pd.read_excel(hierarchyadd, engine='openpyxl')
            #     FCAterms  = GetMapping_E_P(a)

            EtypePropertyList, EtypeList = GetMapping_ET_P(triples)

            # Define the inheritance and the hierarchy for calculating FCAs for both schema and instance level data
            if tag == "schema":
                SonFatherList, FatherSonList, layers, Hierarchy = GetSchemaHierarchy(triples, EtypePropertyList)
                Hierarchy.to_csv("Etypes-hierarchy/%s/%s_Hierarchy.csv" % (topic, name), index=False)
            else:
                Hierarchy = pd.read_csv("Etypes-hierarchy/%s/%s_Hierarchy.csv" % (topic, name))
                SonFatherList, FatherSonList, layers = dict(), dict(), dict()
                for index, row in Hierarchy.iterrows():
                    SonFatherList[row["ETypeName"]] = row["Path"]
                    FatherSonList[row["ETypeName"]] = row["AllSubClasses"]
                    layers[row["ETypeName"]] = row["Layer"]

            # Generate all simple mappings of etype/entity and properties
            FCA = GetCorpus(EtypePropertyList, FatherSonList, SonFatherList,layers)

            # Generate all mappings with consideration of inheritance, as fixed parameter W_E(P)
            FCA_v = GetCorpus_inherited(FCA, SonFatherList, FatherSonList, EtypeList, weight_tag = True)
            # FCA_v = GetCorpus_inherited_large(FCA)
            FCA_h = GetFrequency(FCA_v, weight_tag = True)
            FCA_i = GetEntropy(FCA_h, weight_tag = True)

            FCA_v.to_csv("FCA/%s/%s_FCA-v.csv" % (path.split("/")[-1], name), index=0)
            FCA_h.to_csv("FCA/%s/%s_FCA-h.csv" % (path.split("/")[-1], name), index=0)
            FCA_i.to_csv("FCA/%s/%s_FCA-i.csv" % (path.split("/")[-1], name), index=0)


path = 'triples/Conference'
GenerateFCAs(path,tag = "schema")