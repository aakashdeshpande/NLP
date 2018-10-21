import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed, Dense, Activation
from keras.layers.embeddings import Embedding

def tokenize(data):
    tokenized_data = {'sentences':[], 'tags':[]}
    current_sentence = []
    current_tag = []
    for index, row in data.iterrows():
        if (row['lemma'] == ".") or (row['lemma'] == "!") or (row['lemma'] == "?"):
            tokenized_data['sentences'].append(current_sentence)
            current_sentence = []
            tokenized_data['tags'].append(current_tag)
            current_tag = []
        else:
            current_sentence.append(row['lemma'])
            current_tag.append(row['tag'])
    if current_sentence:
        tokenized_data['sentences'].append(current_sentence)
        tokenized_data['tags'].append(current_tag)
    return tokenized_data

def one_hot_encoding(c, n):
    one_hot_tag = np.zeros(n)
    one_hot_tag[c] = 1
    return one_hot_tag

def encode_tags(row, maxlen, maxtag):
    padded_row = [0] * (maxlen-len(row)) + row
    encoded_row = [one_hot_encoding(tag, maxtag) for tag in padded_row]
    return encoded_row

def accuracy(data):
    tokenized_data = tokenize(data)

    word_set = list(set([word for row in tokenized_data['sentences'] for word in row]))
    word2ind = {word: index for index, word in enumerate(word_set)}
    ind2words = {index: word for index, word in enumerate(word_set)}
    tag_set = list(set([tag for row in tokenized_data['tags'] for tag in row]))
    tag2ind = {tag: (index + 1) for index, tag in enumerate(tag_set)}
    ind2tags = {(index + 1): tag for index, tag in enumerate(tag_set)}
    maxtag = max(tag2ind.values()) + 1

    encoded_data = {}
    encoded_data['sentences'] = [[word2ind[word] for word in row] for row in tokenized_data['sentences']]
    maxlen = max([len(row) for row in encoded_data['sentences']])
    encoded_data['tags'] = [encode_tags([tag2ind[tag] for tag in row], maxlen, maxtag) for row in tokenized_data['tags']]

    X = pad_sequences(encoded_data['sentences'], maxlen = maxlen)
    Y = pad_sequences(encoded_data['tags'], maxlen = maxlen)

    (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=11*32, train_size=45*32, random_state=42)

    max_features = len(word2ind)
    embedding_size = 128
    hidden_size = 32
    out_size = len(tag2ind) + 1

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(out_size)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    batch_size = 32
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=40,
              validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)

    return score