from sklearn.model_selection import train_test_split

import nltk.classify.util
from nltk.classify import MaxentClassifier


def input_features(row):
    row_features = {}
    # row_columns = ['lemma', 'pos', 'shape']
    row_columns = ['lemma']
    for column in row_columns:
        row_features[column + '_' + str(row[column])] = True
    return row_features


def train_features(data):
    data_features = []
    for index, row in data.iterrows():
        data_features.append((input_features(row), str(row['tag'])))
    return data_features


def accuracy(data):
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.40, random_state=42)
    model = MaxentClassifier.train(train_features(data_train), max_iter=2)
    accuracy = nltk.classify.util.accuracy(model, train_features(data_test))
    return accuracy