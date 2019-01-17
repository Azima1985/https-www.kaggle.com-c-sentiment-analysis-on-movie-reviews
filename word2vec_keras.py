import pandas as pd
import spacy
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import GRU,Dropout, BatchNormalization,Dense, GlobalMaxPooling1D, Embedding, Flatten,Bidirectional,LSTM,SpatialDropout1D
from gensim.models import Word2Vec
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K
#Loading the data and the nlp model
nlp=spacy.load('en_core_web_md')
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv',sep='\t')
print('NLP and data loaded...')

#combining train and test data & getting out the sentences
text=pd.concat([train.Phrase,test.Phrase],axis=0)
combined=pd.concat([train,test],axis=0)
sentences=[]
for i in combined.SentenceId.unique():
    sentences.append(combined.loc[combined.SentenceId==i]['Phrase'].iloc[0])

#TRAINING THE WORD2VEC MODEL
#preprocess text
#prepare stopwords
with open('stopwords.txt') as f:
    stopwords=f.readlines()
for i in range(len(stopwords)):
    stopwords[i]=stopwords[i].replace("\'",'').strip()
print('Stopwords loaded...')

def preprocess(x):
    x=nlp(x.lower())
    tokens=[t for t in x]
    #tokens=[t for t in tokens if t.text not in stopwords]
    tokens=[t.text for t in tokens if t.is_punct==False]
    #tokens=[t for t in tokens if len(t)>=3]
    #tokens=[t.lemma_ for t in tokens]
    return tokens
print('Preprocessing...')
processed_text=pd.Series(sentences).apply(preprocess)
print('Finished preprocessing!')
#prepare text for gensim word2vec model
gensim_input=processed_text.tolist()
gensim_input=[lst for lst in gensim_input if len(lst)>0]
w2v=Word2Vec(gensim_input,size=300,min_count=1)
print('Training model...')
w2v.train(gensim_input,total_examples=len(gensim_input),epochs=50)
print('Word2Vec model trained')
print('Number of words: ',len(w2v.wv.vocab))
print('Number of words in the model: ',len(w2v.wv.vocab))# --> number of unique words
#model.wv.index2word[0:5] --> most common word
#model.wv.index2word[-1] --> least common word

#PREPARE DATA FOR NEURAL NETWORK
t = Tokenizer()
t.fit_on_texts(text.tolist())
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(text.tolist())
print("encoded_docs:", len(encoded_docs))
# pad documents to a max length of 56 words (max sentence)
max_length = 56
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print("padded_doc:", len(padded_docs))

#create embedding matrix combining gensim model with keras sequences
embedding_index={word:w2v.wv[word] 
                  if word in w2v.wv else np.zeros((300,)) 
                  for word in t.word_index}

# create a weight matrix for words in docs
embedding_matrix = np.zeros((vocab_size, 300))
for word in t.word_index:
	embedding_matrix[t.word_index.get(word)] = embedding_index.get(word)

#define training, validation and prediction data
xtrain=padded_docs[:140000]
xtest=padded_docs[140000:156060]
xpred=padded_docs[156060:]
ytrain=to_categorical(train.Sentiment.iloc[:140000])
ytest=to_categorical(train.Sentiment.iloc[140000:156060])
print("shape of xtrain:", embedding_matrix.shape)

# define model
model = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix],embeddings_regularizer='l1', input_length=56, trainable=True)
model.add(e)
model.add(Bidirectional(LSTM(50,return_sequences=True),input_shape=(56,300)))
model.add(Flatten())
model.add(Dense(50, activation='elu'))
model.add(Dense(5, activation='softmax'))
#model.load_weights("model.h5")
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(xtrain, ytrain, batch_size=1000,validation_split = 0.05, epochs=6,verbose=2)
epochs = 100
history = model.fit(xtrain, ytrain, batch_size=1000,validation_split = 0.10, epochs=epochs,verbose=2)'''

loss, accuracy = model.evaluate(xtrain, ytrain)
print(f'Training accuracy: {accuracy*100}%')

loss, accuracy = model.evaluate(xtest, ytest)
print(f'Test accuracy: {accuracy*100}%')

preds=model.predict_classes(xpred)
test.to_csv('predictions.csv',columns=['PhraseId','Sentiment'],index=False)
