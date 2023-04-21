import itertools
import yaml
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from owlready2 import get_ontology
from sklearn.metrics import f1_score
import random


def read_Ent(Ent_path):
    classes = []

    EtypeNames = pd.read_csv(Ent_path)["EType"]
    for e in EtypeNames:
        classes.append(e)
    return classes


def read_property(property_path):
    properties = []

    PropertyNames = set(pd.read_csv(property_path).columns[6:])
    for e in PropertyNames:
        properties.append(e)
    return properties

def quick_word_formalization(text):
    text = text.replace("_", "")
    text = text.replace("-", "")
    text = text.strip()
    text = text.lower()

    return text

def get_GTmappings(filename):
    mappings = []

    with open(filename) as f:
        soup = BeautifulSoup(f, 'xml')

    cells = soup.find_all('Cell')

    for cell in cells:
        entity1 = cell.find('entity1').attrs['rdf:resource'].split('#')[1]
        entity2 = cell.find('entity2').attrs['rdf:resource'].split('#')[1]
        mappings.append((entity1, entity2))
        # mappings.append((quick_word_formalization(entity1), quick_word_formalization(entity2)))

    return mappings


def get_dataset(alignment, dataname, PropPair_tag, m_factor=20):
    KG1 = alignment.split('/')[-1].split('.')[0].split('-')[0]
    KG2 = alignment.split('/')[-1].split('.')[0].split('-')[1]

    # read GT mappings from reference data sets (e.g. from OAEI track)
    mappings = get_GTmappings(alignment)
    GTmappings = [tuple(x) for x in mappings]

    # Parse two KGs
    path1 = 'D:\Docs\programs\TrentoLab\OM\Entity type recognition\dataPreparing\FCA/%s/%s_FCA-v.csv' % (dataname,KG1)
    path2 = 'D:\Docs\programs\TrentoLab\OM\Entity type recognition\dataPreparing\FCA/%s/%s_FCA-v.csv' % (dataname,KG2)

    Ent1 = read_Ent(path1)
    Pro1 = read_property(path1)
    Ent2 = read_Ent(path2)
    Pro2 = read_property(path2)

    # all_mappings = []
    ent_pairs = []
    prop_paris = []
    data_unmatched = []

    # Generate pairs of Etypes
    Ent_pairs = list(itertools.product(Ent1, Ent2))
    # print(Ent_pairs)
    for ent_pair in Ent_pairs:
        pair = (ent_pair[0], ent_pair[1])
        if pair in GTmappings:
            # all_mappings.append(pair)
            for i in range(int(m_factor/2)):
                ent_pairs.append((KG1, KG2, pair[0], pair[1], 1, 'Etype'))

        else:
            data_unmatched.append(
                (KG1, KG2, pair[0], pair[1], 0, 'Etype'))

    ent_pairs = ent_pairs + random.sample(data_unmatched, m_factor * len(ent_pairs))

    # Generate pairs of properties
    properties_pairs = list(itertools.product(Pro1, Pro2))
    if PropPair_tag:
        for prop_pair in properties_pairs:
            prop_paris.append(
                (KG1, KG2, prop_pair[0], prop_pair[1], -1, 'Property'))
    else:
        data_unmatched = []
        for prop_pair in properties_pairs:
            pair = (prop_pair[0], prop_pair[1])

            if pair in GTmappings:
                # all_mappings.append(pair)
                prop_paris.append(
                    (KG1, KG2, pair[0], pair[1], 1, 'Property'))
            else:
                data_unmatched.append(
                    (KG1, KG2, pair[0], pair[1], 0, 'Property'))

        prop_paris = prop_paris + random.sample(data_unmatched, m_factor* len(prop_paris))

    data = ent_pairs + prop_paris
    # data = random.shuffle(ent_pairs) + random.shuffle(prop_paris)
    dataset = pd.DataFrame(data, columns=['KG1', 'KG2', 'Label1', 'Label2', 'Match', 'Type'])

    return dataset

