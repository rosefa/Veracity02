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
from sklearn.metrics import confusion_matrix
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
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
import sklearn.metrics as sm

#print(dataTest.head(5))
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
def prepare_model_input(text1,text2,MAX_SEQUENCE_LENGTH=429):
    embeddings_index ={}
    #train = preproces.preprocessing2(text1)
    #test = preproces.preprocessing2(text2)
    tokenizer = Tokenizer()
    #text = np.concatenate((text1, text2), axis=0)
    text = text1+text2
    #tokenizer.fit_on_texts(text2)
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
#dataTest = pd.read_csv('ISOT.csv', encoding= 'unicode_escape')
dataTest = pd.read_csv('FAKESDataset.csv', encoding= 'unicode_escape')
dataTest =sklearn.utils.shuffle(dataTest)
train,test = train_test_split(dataTest,test_size=0.2, shuffle=True)
#valX = preproces.preProcessCorpus(test['text'])
print("GO")
trainX = preproces.preProcessCorpus(train['article_content'])
valX = preproces.preProcessCorpus(test['article_content'])
myTrain_Glove,myVal_Glove,word_index, embedding_matrix = prepare_model_input(trainX,valX)    
modellstmP = preproces.deep_cnn_bilstm(word_index = word_index, embedding_matrix = embedding_matrix)
modellstmP.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
'''
#dataTest = pd.read_csv('ISOT.csv', encoding= 'unicode_escape')
dataTest = pd.read_csv('FAKESDataset.csv', encoding= 'unicode_escape')
dataTest =sklearn.utils.shuffle(dataTest)
#dataTest = dataTest.head(1)
train,test = train_test_split(dataTest,test_size=0.2, shuffle=True)
#valX = preproces.preProcessCorpus(test['text'])
testX = test['article_content']
valY = test['labels']
valX = preproces.preProcessCorpus(test['article_content'])
myTrain_Glove,myVal_Glove,word_index, embedding_matrix = prepare_model_input(valX,valX)
#reconstructed_model = keras.models.load_model("saveBestWeight_bilstmFakesIsot/model0.h5")
reconstructed_model = keras.models.load_model("saveBestWeight_BilstmFakes/best.h5")
#reconstructed = reconstructed_model.predict(myTrain_Glove)
'''
reconstructed = modellstmP.predict(myVal_Glove)
exemple = pd.DataFrame(columns=['Text','Mylabel','LabelPrediction'])
value = np.array(test['labels'], dtype=float)
for i in range(len(reconstructed)):
    if reconstructed[i]<0.5:
        exemple=exemple.append({'Text' : valX[i] , 'Mylabel' :value[i], 'LabelPrediction' :0} , ignore_index=True)
        reconstructed[i]=0
    else:
        exemple=exemple.append({'Text' : valX[i] , 'Mylabel' :value[i], 'LabelPrediction' :1} , ignore_index=True)
        reconstructed[i]=1
exemple.to_csv("TableVeriteFakes2.csv")
np.savetxt('Fatou/valueFakes2.txt', value, delimiter =', ')
np.savetxt('Fatou/valuePreFakes2.txt', reconstructed, delimiter =',')


'''
value = np.array(ValY, dtype=float)

np.savetxt('Fatou/value_1.txt', value, delimiter =', ')
np.savetxt('Fatou/valuePre_1.txt', reconstructed, delimiter =',')

value = np.array(ValY, dtype=float)
reelValue = label['label']
predictValue = label['LabelPrediction']
np.savetxt('Fatou/value.txt', reelValue, delimiter =', ')
np.savetxt('Fatou/valuePre.txt', predictValue, delimiter =',')
#print(reelValue)
#print(predictValue)
matrix_seuil = confusion_matrix(reelValue, predictValue)

print(matrix_seuil)
print (sm.accuracy_score(reelValue, predictValue))
print (sm.precision_score(reelValue, predictValue))
print (sm.recall_score(reelValue, predictValue))
print (sm.f1_score(reelValue, predictValue))

reconstructed_model = keras.models.load_model("saveBestWeight_bilstmFakesIsot/model0.h5")
reconstructed = reconstructed_model.predict(myTrain_Glove)
label =1
for i in range(len(valX)):
    if reconstructed[i]<0.5:
        label =0
    if reconstructed[i]>0.5:
        label =1
    exemple=exemple.append({'Text' : valX[i] , 'Prediction' :reconstructed[i], 'LabelPredict' :label} , ignore_index=True)
    print(reconstructed[i])

exemple.to_csv("exempleAppliPredictISOT_2.csv")'''
