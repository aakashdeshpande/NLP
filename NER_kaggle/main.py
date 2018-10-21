import pandas as pd
#from model.maxent import accuracy
from model.rnn import accuracy

data = pd.read_csv('input/ner.csv', encoding='utf-8')

labels = data.iloc[:, -1]
# data = data.iloc[:,1:-1]
# print(data.columns)
# data, labels = np.arange(10).reshape((5, 2)), range(5))


# print(data_train.size/data_test.size)


# tokenizer = Tokenizer(filters='')
# data['lemma'] = data['lemma'].apply(lambda x: str(x))
# print(data.loc[type(data['lemma']) is float])
# tokenizer.fit_on_texts(data['lemma'].values)
# X = tokenizer.texts_to_sequences(data['lemma'].values)
# index_word = {v: k for k, v in tokenizer.word_index.items()}
# print(index_word.get(1501))
# print(index_word.get(151))

accuracy = accuracy(data)
print(accuracy)
