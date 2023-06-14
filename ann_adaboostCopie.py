import preprocessing as preproces
import spacy
import pysbd
from keras.layers.normalization.batch_normalization import BatchNormalization
import wget
import nltk
from math import *
import numpy as np
from numpy import asarray
from numpy import zeros
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
import inflect
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, Layer
from keras.layers import Conv1D
from keras.layers import MaxPooling1D, GlobalMaxPool1D, GlobalMaxPooling1D, Dropout
from keras.layers import LSTM, GRU
from keras.layers import Bidirectional,Flatten
import numpy as np
import pandas as pd
import re
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
import tensorflow as tf 
import io
from tqdm import tqdm
#import chardet
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import tensorflow_hub as hub
import statistics
import unicodedata
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer,PorterStemmer
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
#import tensorflow_decision_forests as tfdf
#import scikeras
#from scikeras.wrappers import KerasClassifier, KerasRegressor
#from sklearn.ensemble import AdaBoostClassifier
#nltk.download('averaged_perceptron_tagger')
scaler = MinMaxScaler(feature_range=(-1,1))


dataf1 = pd.read_csv('Pasvrai-1.csv', encoding= 'unicode_escape')
dataf2 = pd.read_csv('Pasvrai-2.csv', encoding= 'unicode_escape')
dataf3 = pd.read_csv('Pasvrai-3.csv', encoding= 'unicode_escape')
datav1 = pd.read_csv('True-1.csv', encoding= 'unicode_escape')
datav2 = pd.read_csv('True-2.csv', encoding= 'unicode_escape')
datav3 = pd.read_csv('True-3.csv', encoding= 'unicode_escape')
datav4 = pd.read_csv('True-4.csv', encoding= 'unicode_escape')

neg =[]
i=0
while i<len(dataf1):
  neg.append(0)
  i=i+1
dataf1['label']=neg
i=0
neg =[]
while i<len(dataf2):
  neg.append(0)
  i=i+1
dataf2['label']=neg
i=0
neg =[]
while i<len(dataf3):
  neg.append(0)
  i=i+1
dataf3['label']=neg
pos =[]
i=0
while i<len(datav1):
  pos.append(1)
  i=i+1
datav1['label']=pos
pos =[]
i=0
while i<len(datav2):
  pos.append(1)
  i=i+1
datav2['label']=pos
pos =[]
i=0
while i<len(datav3):
  pos.append(1)
  i=i+1
datav3['label']=pos
pos =[]
i=0
while i<len(datav4):
  pos.append(1)
  i=i+1
datav4['label']=pos
data = pd.concat([dataf1,dataf2,dataf3,datav1,datav2,datav3,datav4], axis=0)
#print(list(data.columns))
data =sklearn.utils.shuffle(data)
dataTest = pd.read_csv('FAKESDataset.csv', encoding= 'unicode_escape')
dataTest =sklearn.utils.shuffle(dataTest)
#train_data = pd.DataFrame(dataset)

def plot_graphs(history,string,nom):
    fig, ax = plt.subplots()
    if(string=='accuracy'):
        ax.plot(history.history[string],ls='-', color='b',linewidth=2)
        ax.plot(history.history['val_'+string],ls='-', color='r',linewidth=2)
    if(string=='loss'):
        ax.plot(history.history[string],ls=':', color='b',linewidth=2)
        ax.plot(history.history['val_'+string],ls=':', color='r',linewidth=2)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(string)
    ax.set_title('FA-KES')
    ax.grid(linestyle = '--', linewidth = 0.8)
    if (string=='accuracy'):
        ax.legend(['Train_Acc','Validation_Acc'])
    if (string=='loss'):
        ax.legend(['Train_Loss','Validation_Loss'])
    fig.savefig(nom+'FAKES.png', transparent=False, bbox_inches="tight")

def prepare_model_input(text1,text2,MAX_SEQUENCE_LENGTH=429):
    embeddings_index ={}
    tokenizer = Tokenizer()
    text = text1+text2
    tokenizer.fit_on_texts(text)
    tokenizer.word_index['<PAD>'] = 0
    sequencesVal = tokenizer.texts_to_sequences(text2)
    sequencesText = tokenizer.texts_to_sequences(text1)
    val_Glove = tf.keras.preprocessing.sequence.pad_sequences(sequencesVal, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    text_Glove = tf.keras.preprocessing.sequence.pad_sequences(sequencesText, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    word_index = tokenizer.word_index
    with open("glove.6B.100d.txt") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    nb_words = len(word_index)
    embedding_matrix = np.zeros((nb_words+1,100))
    count_found = nb_words
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] =  embedding_vector
    return (text_Glove,val_Glove,word_index, embedding_matrix)


kf = StratifiedKFold(n_splits=5,shuffle=True)
X1 = dataTest['article_content']
Y1 = dataTest['labels']
X = data['text']
Y = data['label']


for i in range (2):
    print("*************"+str(i)+"***********************")
    fold_var = 1
    trainX = preproces.preProcessCorpus(X)
    valX = preproces.preProcessCorpus(X1)
    myTrain_Glove,myVal_Glove,word_index, embedding_matrix = prepare_model_input(trainX,valX)
    modellstmP = preproces.deep_cnn_bilstm(word_index = word_index, embedding_matrix = embedding_matrix)
    modellstmP.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    callbacks = [
            keras.callbacks.ModelCheckpoint("saveBestWeight_bilstmFakesIsot/modelOtre"+str(i)+".h5",monitor='val_accuracy',verbose=0,save_best_only=True,mode='max')
        ]
    modellstmP.fit(myTrain_Glove,Y,validation_data=(myVal_Glove,Y1), epochs=10, batch_size=64, callbacks=callbacks,verbose=1)
    tf.keras.backend.clear_session()
