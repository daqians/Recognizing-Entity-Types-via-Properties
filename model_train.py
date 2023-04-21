import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
import argparse
from sklearn.neural_network import MLPClassifier



def training(opt):
    # Generate training set and testing set
    features = pd.read_csv("TrainingSets/%s_features.csv" % opt.dataset)
        # "D:\Docs\programs\TrentoLab\OM\Entity type recognition\TrainingSets/%s_features.csv" % opt.dataset)
    train_features, test_features = train_test_split(features, test_size=0.3, random_state=42)

    # Or use one track as training set and another as testing set
    # train_features = pd.read_csv("D:\Docs\programs\TrentoLab\OM\Entity type recognition\TrainingSets/%s_features.csv" % opt.dataset_train)
    # test_features = pd.read_csv("D:\Docs\programs\TrentoLab\OM\Entity type recognition\TrainingSets/%s_features.csv" % opt.dataset)

    """    
    Choosing the features for ETR, 
    we have string-based features: Ngram1, Longest_com_sub, Levenshtein;
    semantic-based features: Wordnet_sim, Word2vec_sim;
    and property-based features: Sim_V, Sim_H, Sim_I.
    """
    X_train = train_features.loc[:, 'Ngram1':'Sim_I']
    # X_train = train_features.loc[:, 'Ngram':'Sim_V']
    # X_train = train_features.loc[:, 'Sim_V':'Sim_H']
    # X_train = train_features.loc[:, 'Ngram1':'Word2vec_sim']
    # X_train.drop(['Sim_V'], axis=1)

    X_train = X_train.fillna(value=0)
    Y_train = train_features['Match']

    X_test = test_features.loc[:, 'Ngram1':'Sim_I']
    # X_test = test_features.loc[:, 'Ngram':'Sim_V']
    # X_test = test_features.loc[:, 'Sim_V':'Sim_H']
    # X_test = test_features.loc[:, 'Ngram1':'Word2vec_sim']
    # X_test.drop(['Sim_V'], axis=1)

    X_test = X_test.fillna(value=0)
    Y_test = test_features['Match']

    # Choosing ML models
    if opt.model == 'DecisionTree':
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)

    elif opt.model == 'RandomForest':
        model = RandomForestClassifier(n_estimators=500,
                                       max_features='sqrt', max_depth=3,
                                       random_state=42)
        model.fit(X_train, Y_train)

    elif opt.model == 'LogisticRegression':
        model = LogisticRegression(penalty='l2', C=7.742637,
                                   class_weight=None)
        model.fit(X_train, Y_train)

    elif opt.model == 'XGBoost':
        # Dtrain = xgb.DMatrix( X_train, label=Y_train)
        # Dtest = xgb.DMatrix( X_test, label=Y_test)
        #
        # param = {'max_depth': 5, 'eta': 0.1, 'silent': 1, 'subsample': 0.7, 'colsample_bytree': 0.7,
        #          'objective': 'binary:logistic'}
        #
        # watchlist = [(Dtest, 'eval'), (Dtrain, 'train')]
        # num_round = 10
        # model = xgb.train(param, Dtrain, num_round, watchlist)

        model = xgb.XGBClassifier(n_estimators=20, \
                                           max_depth=5, \
                                           learning_rate=0.1, \
                                           subsample=0.7, \
                                           colsample_bytree=0.7, \
                                           eval_metric='error')
        model.fit(X_train, Y_train)

    elif opt.model == 'ANN':
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        model.fit(X_train, Y_train)

    elif opt.model == 'SGDClassifier':
        # model = SGDClassifier(loss='log')
        model = SGDClassifier()
        model.partial_fit(X_train, Y_train, classes=np.array([0, 1]))



    y_prob = model.predict(X_test)

    # print(Y_test,y_prob)

    # aa = model.predict_proba(X_test)
    # aa_proba = []
    #
    # for a in aa:
    #     if a[1]>0.5:
    #         aa_proba.append(1)
    #     else:
    #         aa_proba.append(0)

    pred_mappings, true_mappings, correct_mappings = bestMapping(y_prob, test_features)

    #     from sklearn.metrics import confusion_matrix, classification_report
    #     print(classification_report(Y_test, y_prob))
    #
    #     cm = metrics.confusion_matrix(Y_test, y_prob)
    #     print('Confusion Matrix: \n', cm)

    Accuracy_present(pred_mappings, true_mappings, opt.dataset)

    # SeparateResults(pred_mappings,true_mappings)

    true_mappings.to_csv('results/%s/true.csv' %opt.dataset)
    pred_mappings.to_csv('results/%s/pred.csv' %opt.dataset)
    correct_mappings.to_csv('results/%s/correct.csv' %opt.dataset)


def bestMapping(y_prob, test_features):
    test_features['Predict'] = y_prob

    pred_mappings = test_features[(test_features['Predict'] == 1)]
    true_mappings = test_features[(test_features['Match'] == 1)]
    correct_mappings = test_features[(test_features['Match'] == 1) & (test_features['Predict'] == 1)]
    # pred_mappings.drop(columns=['Match'],axis = 1,inplace=True)

    return pred_mappings, true_mappings, correct_mappings


def Accuracy_present(pred_mappings, true_mappings, name):
    precison = true_mappings[(true_mappings["Predict"] == 1)].shape[0] / true_mappings.shape[0]
    recall = pred_mappings[(pred_mappings["Match"] == 1)].shape[0] / pred_mappings.shape[0]
    F1 = 2 * precison * recall / (precison + recall)
    F05 = 1.25 * precison * recall / (1.25 * precison + recall)
    F2 = 5 * precison * recall / (5 * precison + recall)

    print('%s precison: %f' % (name, precison))
    print('%s recall: %f' % (name, recall))
    print("%s F1 score: %f" % (name, F1))
    # return precison,recall,F1

def SeparateResults(pred_mappings, true_mappings):
    preds = dict()
    pred = pd.DataFrame(
        columns=['Ontology1', 'Ontology2', 'Name1', 'Name2', 'Match', 'Type', 'Ngram1_Entity', 'Longest_com_sub_Entity',
                 'Levenshtein_Entity',
                 'Wordnet_sim_Entity', 'Word2vec_sim_Entity', 'Etype_similarity-v', 'Etype_similarity-h', 'Predict'])
    true = pd.DataFrame(
        columns=['Ontology1', 'Ontology2', 'Name1', 'Name2', 'Match', 'Type', 'Ngram1_Entity', 'Longest_com_sub_Entity',
                 'Levenshtein_Entity',
                 'Wordnet_sim_Entity', 'Word2vec_sim_Entity', 'Etype_similarity-v', 'Etype_similarity-h', 'Predict'])

    ONT = ('cmt', 'confof')
    for index, row in pred_mappings.iterrows():
        ont1 = str(row['Ontology1']).split('/')[-1]
        ont2 = str(row['Ontology2']).split('/')[-1]

        if (ont1, ont2) != ONT:
            preds[ONT] = pred
            ONT = (ont1, ont2)
            pred = pd.DataFrame(columns=['Ontology1', 'Ontology2', 'Name1', 'Name2', 'Match', 'Type', 'Ngram1_Entity',
                                         'Longest_com_sub_Entity', 'Levenshtein_Entity',
                                         'Wordnet_sim_Entity', 'Word2vec_sim_Entity', 'Etype_similarity-v',
                                         'Etype_similarity-h', 'Predict'])
        else:
            pred = pred.append(row)

    ONT = ('cmt', 'confof')
    for index, row in true_mappings.iterrows():
        ont1 = str(row['Ontology1']).split('/')[-1]
        ont2 = str(row['Ontology2']).split('/')[-1]

        if (ont1, ont2) != ONT:
            name = str(ONT[0]) + '-' + str(ONT[1])
            Accuracy_present(pred, true, name)

            ONT = (ont1, ont2)
            true = pd.DataFrame(columns=['Ontology1', 'Ontology2', 'Name1', 'Name2', 'Match', 'Type', 'Ngram1_Entity',
                                         'Longest_com_sub_Entity', 'Levenshtein_Entity',
                                         'Wordnet_sim_Entity', 'Word2vec_sim_Entity', 'Etype_similarity-v',
                                         'Etype_similarity-h', 'Predict'])
        else:
            true = true.append(row)


if __name__ == '__main__':
    ##### arguments settings #####
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='ANN', help="the ML model used to train and test, can be"
                                                                           "SGDClassifier, DecisionTree, "
                                                                           "RandomForest, LogisticRegression, XGBoost")
    parser.add_argument("--dataset", type=str, default='Conference', help="The training dataset, can be General, "
                                                                          "BiblioTrack, Conference")
    parser.add_argument("--dataset_train", type=str, default='Conference', help="An optional parameter for generating "
                                                                                "the training dataset, can be General,"
                                                                                " BiblioTrack, Conference")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    opt = parser.parse_args()

    # print("training the ETR model on %s Track" % opt.dataset_train)
    print("testing the ETR model on %s Track" % opt.dataset)
    print("Training by %s model" % opt.model)

    ##### Run training process #####
    training(opt)
