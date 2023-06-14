import preprocessing as preproces
import spacy
import pysbd
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
#import tensorflow_decision_forests as tfdf
from keras.layers.normalization.batch_normalization import BatchNormalization
import wget
#nltk.download('omw-1.4')
import nltk
from numpy import asarray
from numpy import zeros
from sklearn.preprocessing import LabelEncoder
import inflect
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
#import tensorflow_decision_forests as tfdf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, Layer
from keras.layers import Conv1D
from keras.layers import MaxPooling1D, GlobalMaxPool1D, GlobalMaxPooling1D, Dropout
from keras.layers import LSTM, GRU
from keras.layers import Bidirectional,Flatten
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow import keras
#import tensorflow_decision_forests as tfdf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
#from keras.wrappers.scikit_learn import KerasRegressor
#from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf 
import io
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
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
#nltk.download('averaged_perceptron_tagger')


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
dataTest = pd.read_csv('FAKESDataset.csv', encoding= 'unicode_escape')

def builModel ():
  model = Sequential()
  model.add(layers.Conv1D(128, 5,activation='relu',input_shape=(512, 1)))
  model.add(layers.MaxPooling1D(2))
  model.add(LSTM(32))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
  return model

def build_bilstm(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100):
    optimizer = tf.keras.optimizers.Adam()
    input = Input(shape=(300,), dtype='int32')
    embedding_matrix = np.random.random((len(word_index)+1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=300,trainable=True)(input)
    model = Conv1D(128, 5,activation='relu')(embedding_layer)
    model = MaxPooling1D(2)(model)
    model = LSTM(32)(model)
    lastLayer = Dense(1,activation='sigmoid')(model)
    model = keras.Model(inputs=input,outputs=lastLayer)
    nn_without_head = tf.keras.models.Model(inputs=model.inputs, outputs=lastLayer)
    #df_and_nn_model = tfdf.keras.RandomForestModel(preprocessing=nn_without_head,num_trees=300)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    return (model)
    
def plot_graphs(history1,history2,history3,history4, string,nom):
    fig, ax = plt.subplots()
    ax.plot(history1.history[string],ls='-', color='b')
    ax.plot(history2.history[string],ls='-', color='r')
    ax.plot(history3.history[string],ls='-', color='g')
    ax.plot(history4.history[string],ls='-', color='k')
    #plt.plot(history.history['val_'+string])
    ax.set_xlabel("Epochs")
    ax.set_ylabel(string)
    #plt.legend([string, 'val_'+string])
    ax.legend(['CnnLstm','DcnnLstm','CnnMtl','CnnBilstm'])
    fig.savefig(nom+'.png', transparent=False, bbox_inches="tight")
    #plt.show()
  
def prepare_model_input(X,MAX_NB_WORDS=45000,MAX_SEQUENCE_LENGTH=300):
    #tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_Glove = pad_sequences(sequences, maxlen=300)
    word_index = tokenizer.word_index
    
    embeddings_dict = {}
    f = open("glove.6B.100d.txt", encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass
        embeddings_dict[word] = coefs
    f.close()
    return (X_Glove, word_index, embeddings_dict)
#def evaluation_model()


'''**************CROSS VALIDATION********************'''
kfold = StratifiedKFold(n_splits=10, shuffle=True)
cvscores = []
train,test = train_test_split(dataTest,test_size=0.2, shuffle=True)
trainX = train['article_content']
trainY = train['labels']
testX = test['article_content']
testY = test['labels']
trainX = preproces.preprocessing(trainX)
testX = preproces.preprocessing(testX)
Xpre = preproces.preprocessing(dataTest['article_content'])
myData_Glove,word_index, embeddings_dict = prepare_model_input(Xpre)
#myData_Glove_train,word_index_train, embeddings_dict_train = prepare_model_input(trainX)
#myData_Glove_test,word_index_test, embeddings_dict_test = prepare_model_input(testX)
#model,df_and_nn_model = build_bilstm(word_index=word_index, embeddings_dict=embeddings_dict)
modelCnnLstm = preproces.cnn_lstm(word_index=word_index, embeddings_dict=embeddings_dict)
modelDcnnLstm = preproces.dcnn_lstm(word_index=word_index, embeddings_dict=embeddings_dict)
modelCnnMtl = preproces.cnn_mtl(word_index=word_index, embeddings_dict=embeddings_dict)
modelCnnBilstm = preproces.cnn_bilstm(word_index=word_index, embeddings_dict=embeddings_dict)

#n_folds = 10
#kfold = KFold(n_folds, True, 1)
# cross validation estimation of performance
scoresAccCnnLstm= list()
scoresPrecCnnLstm= list()
scoresRapelCnnLstm= list()
scoresAccDcnnLstm= list()
scoresPrecDcnnLstm= list()
scoresRapelDcnnLstm= list()
scoresAccCnnMtl= list()
scoresPrecCnnMtl= list()
scoresRapelCnnMtl= list()
scoresAccCnnBilstm= list()
scoresPrecCnnBilstm= list()
scoresRapelCnnBilstm= list()
scoresAccTFR= list()
Y = dataTest['labels']
i=1
for train, test in kfold.split(myData_Glove,Y):
	# select samples
	trainX, trainy = myData_Glove[train], Y[train]
	testX, testy = myData_Glove[test], Y[test]
	#myData_Glove_train,word_index_train, embeddings_dict_train = prepare_model_input(trainX)
	#myData_Glove_test,word_index_test, embeddings_dict_test = prepare_model_input(testX)
	#model,df_and_nn_model = build_bilstm(word_index=word_index, embeddings_dict=embeddings_dict)
	historyCnnLstm = modelCnnLstm.fit(trainX, trainy,validation_data=(testX, testy), epochs=100, batch_size=64, verbose=0)
	historyDcnnLstm = modelDcnnLstm.fit(trainX, trainy,validation_data=(testX, testy), epochs=100, batch_size=64, verbose=0)
	historyCnnMtl = modelCnnMtl.fit(trainX, trainy,validation_data=(testX, testy), epochs=100, batch_size=64, verbose=0)
	historyCnnBilstm = modelCnnBilstm.fit(trainX, trainy,validation_data=(testX, testy), epochs=100, batch_size=64, verbose=0)
	plot_graphs(historyCnnLstm,historyDcnnLstm,historyCnnMtl,historyCnnBilstm, string='accuracy',nom='TrainAccuracy_'+str(i))
	plot_graphs(historyCnnLstm,historyDcnnLstm,historyCnnMtl,historyCnnBilstm, string='val_accuracy',nom='ValidationAccuracy_'+str(i))
	plot_graphs(historyCnnLstm,historyDcnnLstm,historyCnnMtl,historyCnnBilstm, string='loss',nom='TrainLoss_'+str(i))
	plot_graphs(historyCnnLstm,historyDcnnLstm,historyCnnMtl,historyCnnBilstm, string='val_loss',nom='ValidationLoss_'+str(i))
	test_accCnnLstm = modelCnnLstm.evaluate(testX, testy)
	test_accDcnnLstm = modelDcnnLstm.evaluate(testX, testy)
	test_accCnnMtl = modelCnnMtl.evaluate(testX, testy)
	test_accCnnBilstm = modelCnnBilstm.evaluate(testX, testy)
	#df_and_nn_model.compile(metrics=["accuracy"])
	#df_and_nn_model.fit(trainX, trainy)
	#test_acc_TFR=df_and_nn_model.evaluate(testX, testy)
	scoresAccCnnLstm.append(test_accCnnLstm[1])
	scoresPrecCnnLstm.append(test_accCnnLstm[2])
	scoresRapelCnnLstm.append(test_accCnnLstm[3])
	#**********************
	scoresAccDcnnLstm.append(test_accDcnnLstm[1])
	scoresPrecDcnnLstm.append(test_accDcnnLstm[2])
	scoresRapelDcnnLstm.append(test_accDcnnLstm[3])
	#**********************
	scoresAccCnnMtl.append(test_accCnnMtl[1])
	scoresPrecCnnMtl.append(test_accCnnMtl[2])
	scoresRapelCnnMtl.append(test_accCnnMtl[3])
	#**********************
	scoresAccCnnBilstm.append(test_accCnnBilstm[1])
	scoresPrecCnnBilstm.append(test_accCnnBilstm[2])
	scoresRapelCnnBilstm.append(test_accCnnBilstm[3])
	#**********************
	#plot_graphs(history, 'accuracy')
	#plot_graphs(history, 'loss')
	print(i)
	i=i+1
# summarize expected performance
print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scoresAccCnnLstm), np.std(scoresAccCnnLstm)))
print('Estimated Precision %.3f (%.3f)' % (np.mean(scoresPrecCnnLstm), np.std(scoresPrecCnnLstm)))
print('Estimated Rappel %.3f (%.3f)' % (np.mean(scoresRapelCnnLstm), np.std(scoresRapelCnnLstm)))
#print('Estimated scoresAccTFR %.3f (%.3f)' % (np.mean(scoresAccTFR), np.std(scoresAccTFR)))
print('************************************')
print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scoresAccDcnnLstm), np.std(scoresAccDcnnLstm)))
print('Estimated Precision %.3f (%.3f)' % (np.mean(scoresPrecDcnnLstm), np.std(scoresPrecDcnnLstm)))
print('Estimated Rappel %.3f (%.3f)' % (np.mean(scoresRapelDcnnLstm), np.std(scoresRapelDcnnLstm)))
#print('Estimated scoresAccTFR %.3f (%.3f)' % (np.mean(scoresAccTFR), np.std(scoresAccTFR)))
print('************************************')
print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scoresAccCnnMtl), np.std(scoresAccCnnMtl)))
print('Estimated Precision %.3f (%.3f)' % (np.mean(scoresPrecCnnMtl), np.std(scoresPrecCnnMtl)))
print('Estimated Rappel %.3f (%.3f)' % (np.mean(scoresRapelCnnMtl), np.std(scoresRapelCnnMtl)))
#print('Estimated scoresAccTFR %.3f (%.3f)' % (np.mean(scoresAccTFR), np.std(scoresAccTFR)))
print('************************************')
print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scoresAccCnnBilstm), np.std(scoresAccCnnBilstm)))
print('Estimated Precision %.3f (%.3f)' % (np.mean(scoresPrecCnnBilstm), np.std(scoresPrecCnnBilstm)))
print('Estimated Rappel %.3f (%.3f)' % (np.mean(scoresRapelCnnBilstm), np.std(scoresRapelCnnBilstm)))
#print('Estimated scoresAccTFR %.3f (%.3f)' % (np.mean(scoresAccTFR), np.std(scoresAccTFR)))

#model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
'''history = model.fit(myData_Glove_train, trainY,validation_data=(myData_Glove_test, testY), epochs=10, batch_size=64, verbose=1)
print("Evaluation :", model.evaluate(myData_Glove_test, testY))
df_and_nn_model.compile(metrics=["accuracy"])
df_and_nn_model.fit(myData_Glove_train, trainY)
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
print("Evaluation:", df_and_nn_model.evaluate(myData_Glove_test, testY))'''
'''for train, test in kfold.split(myData_Glove,Y):
  #model = KerasClassifier(build_bilstm, word_index=word_index, embeddings_dict=embeddings_dict,verbose=0)
  model = build_bilstm(word_index=word_index, embeddings_dict=embeddings_dict)
  #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
  history=model.fit(myData_Glove[train], Y[train], validation_data=(myData_Glove[test], Y[test]),epochs=10, batch_size=64, verbose=0)
# evaluate the model
  scores = model.evaluate(myData_Glove[test], Y[test], verbose=1)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
  print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
  cvscores.append(scores[1] * 100)
  plot_graphs(history, 'accuracy')
  plot_graphs(history, 'loss')
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))'''
