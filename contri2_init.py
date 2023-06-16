import Nettoyage as net
#import nltk
import pandas as pd
import numpy as np 
import cv2 as cv
#from fastai.imports import *
import os, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import timeit
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.constraints import Constraint

import tensorflow as tf
from tensorflow import keras
import keras.utils as image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.models import load_model

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from keras.layers import Bidirectional,LSTM, Add,GRU,MaxPooling1D, GlobalMaxPool1D, GlobalMaxPooling1D, Dropout,Conv1D,Embedding,Flatten, Input, Layer,GlobalAveragePooling1D,Activation,Lambda,LayerNormalization, Concatenate, Average,AlphaDropout,Reshape, multiply

#import contractions
#from bs4 import BeautifulSoup
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
#from nltk import word_tokenize, sent_tokenize, pos_tag
#from nltk.corpus import stopwords
#from nltk.stem import LancasterStemmer, WordNetLemmatizer,PorterStemmer
from tensorflow.keras.layers import TextVectorization
#import tqdm
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import roc_curve, auc
import spacy
from scipy import stats
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
print('DEBUT..............')


#print(dataTest.head(5))
#dataText = pd.read_csv('MediaEvalData/Eval2015_TestSet.csv',delimiter=";", encoding= 'unicode_escape')
#dataText = pd.read_csv('MediaEvalData/Eval2015_DevSet1.csv',delimiter=";", encoding= 'unicode_escape')
dataText = pd.read_csv('MediaEvalData/DevSetOtre.txt', sep="\t",header=0, engine='python',encoding= 'raw_unicode_escape')

print("######################## DATA IMAGE SHAPE ###################")
#print(dataText.shape)
#print(dataText.head())

'''
image_test1 = 'MediaEvalData/TestSetImages/Garissa_Attack/'
image_test2 = 'MediaEvalData/TestSetImages/Nepal_earthquake'
image_test3 = 'MediaEvalData/TestSetImages/Samurai_ghost'
image_test4 = 'MediaEvalData/TestSetImages/Solar_eclipse'
image_test4 = 'MediaEvalData/TestSetImages/Syrian_boy'
image_test4 = 'MediaEvalData/TestSetImages/Varoufakis_zdf'
for filename in glob.glob(image_test1+'*.jpg'):
    img_color=cv2.imread(filename,-1)
    plt.imshow(img_color)
    plt.axis("off")
    plt.show()
'''
height = 224
width = 224
dim = (width, height)

x  = 'MediaEvalData/DevSetImages'
dir_path = 'MediaEvalData/DevSetImages/**/**'
#res est l'ensemble des dossiers contenant les images
res = glob.glob(dir_path)
dataImage = pd.DataFrame(columns=['nomImage','image','label'])
textListe = []
imageListe = []
labelImage = []
labelText = []
#Former un dataframe les images et le label
i=0
while i < len(res):
    if(res[i][-5:]=='fakes'):
        chem = glob.glob(res[i]+'/**')
        j=0
        while j <len(chem):
            img = Image.open(chem[j]).convert('RGB')
            imgResize = img.resize((224,224),Image.ANTIALIAS)
            #imgResize = imgResize/255
            #imgResize = imgResize.astype(np.float32)
            chemin = chem[j].split("/")
            imageNom = chemin[len(chemin)-1]
            nb = len(imageNom)-4
            dataImage.loc[len(dataImage)] = [imageNom[:nb],np.array(imgResize),0]
            j=j+1
    if(res[i][-5:]=='reals'):
        chem = glob.glob(res[i]+'/**')
        j=0
        while j < len(chem):
            img = Image.open(chem[j]).convert('RGB')
            imgResize = img.resize((224,224),Image.ANTIALIAS)
            #imgResize = imgResize/255
            #imgResize = imgResize.astype(np.float32)
            chemin = chem[j].split("/")
            imageNom = chemin[len(chemin)-1]
            nb = len(imageNom)-4
            dataImage.loc[len(dataImage)] = [imageNom[:nb],np.array(imgResize),1]
            j=j+1
    if(res[i][-4:]=='.jpg'or res[i][-4:]=='.png' ) :
        #img=mpimg.imread(res[i])
        img = Image.open(res[i]).convert('RGB')
        imgResize = img.resize((224,224),Image.ANTIALIAS)
        chemin = res[i].split("/")
        imageNom = chemin[len(chemin)-1]
        nb = len(imageNom)-4
        dataImage.loc[len(dataImage)] = [imageNom[:nb],np.array(imgResize),0]
    i=i+1
print("######################## DATA IMAGE SHAPE ###################")
#print(dataImage.shape)
#print(dataImage.head())

dataImageText = pd.DataFrame(columns=['text','nomImage','image','label'])
k=0
i=0
while i <len(dataText):
    equal = 'non'
    if(dataText.loc[i,'label']=='fake'or dataText.loc[i,'label']=='real'):
        j=0
        while j < len(dataImage):
            if(dataText.loc[i,'imageId(s)']==dataImage.loc[j,"nomImage"]):
                imageListe.append(dataImage.loc[j,"image"])
                labelImage.append(dataImage.loc[j,"label"])
                tweetClean = net.clean(dataText.loc[i,'tweetText'])
                textListe.append(str(tweetClean))
                labelText.append(dataText.loc[i,'label'])
                equal = 'oui'
            j=j+1
    i=i+1
    
imageListe = np.array(imageListe)
#textListe = np.array(textListe)
# Encode y text data in numeric
encoder = LabelEncoder()
encoder.fit(labelText)
y = encoder.transform(labelText)



#x_train, x_test,Im_train, Im_test, y_train, y_test = train_test_split(textListe,imageListe,y, test_size = 0.2, random_state = 42)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#print("DOWNLOAD stopwords and ...")
#nltk.download('stopwords')
#nltk.download('punkt')
#print("DOWNLOAD stopwords and OK")
'''
def preProcessCorpus(docs):
  norm_docs = []
  stop_words = set(stopwords.words('english'))
  i=0
  for doc in tqdm.tqdm(docs):
    #doc = strip_html_tags(doc)
    doc = re.sub(r'^http?:\/\/.*[\r\n]*', '', doc,flags=re.MULTILINE)
    doc = re.sub('\&lt\;.*?\&gt\;', '', doc)
    doc = doc.lower()
    doc = doc.translate(doc.maketrans("\n\t", "  "))
    #doc = remove_accented_chars(doc)
    doc = contractions.fix(doc)
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = re.sub(' +', ' ', doc)
    doc = re.sub(r'[0-9]+', '', doc)
    word_tokens = word_tokenize(doc)
    doc = [w for w in word_tokens if not w in stop_words]
    doc = ' '.join([x for x in doc])
    doc = doc.strip()  
    norm_docs.append(doc)
    i=i+1
  return norm_docs
'''

def prepare_model_input(text1,MAX_SEQUENCE_LENGTH=100):
    embeddings_index ={}
    tokenizer = Tokenizer()
    #text = text1+text2
    text = text1
    tokenizer.fit_on_texts(text)
    tokenizer.word_index['<PAD>'] = 0
    #sequencesVal = tokenizer.texts_to_sequences(text2)
    sequencesText = tokenizer.texts_to_sequences(text1)
    #val_Glove = tf.keras.preprocessing.sequence.pad_sequences(sequencesVal, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
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
    return (text_Glove,word_index, embedding_matrix)
def clean(doc):
    doc = [token.lemma_.lower() for token in doc if (not token.is_punct) and (not token.is_space) and (not token.like_url) and (not token.is_stop) and len(token) > 1]
    doc = ' '.join([x for x in doc])
    return doc
docsClean = []
i=0
print("DEBUT Traitement de text ...")
while i<len(textListe):
    docs = nlp(textListe[i])
    docsClean.append(clean(docs))
    print(i)
    i=i+1
print("FIN Traitement de text")
#trainX = preProcessCorpus(x_train)
#valX = preProcessCorpus(x_test)
#myTrain_Glove,myVal_Glove,word_index, embedding_matrix = prepare_model_input(x_train, x_test)
myTrain_Glove, word_index, embedding_matrix = prepare_model_input(docsClean)
#K.clear_session()

##########INITIALISATION DES FILTRES###########
###création d'une matrice de valeurs aleatoires de type float###
'''
n=5
np.random.rand2 = lambda *args, dtype=np.float32: np.random.rand(*args).astype(dtype)
f1 = np.random.rand2(n,n)
custom_weights = np.array((f1, f1, f1))
f2 = custom_weights.transpose(1, 2, 0)
custom_weights = np.array((f2,f2,f2,f2,f2,f2,f2,f2,f2,f2,f2,f2))
custom_weights = custom_weights.transpose(1, 2, 3, 0)
'''
##########DEFINITION DE LA CONTRAINTE SUR LES FILTRES###########
'''
class CustomConstraint(Constraint):
    def __init__(self, custom_weights):
        self.custom_weights = tf.Variable(custom_weights)
    def __call__(self, weights):
        output = self.custom_weights
        row_index = 2
        col_index = 2
        new_value = 0
        output[row_index,col_index,:,:].assign(new_value)
        som = tf.keras.backend.sum(output)
        sum_without_center1 = 1/som
        newMatrix = output*sum_without_center1
        output.assign(newMatrix)
        new_value = -1
        output[row_index,col_index,:,:].assign(new_value)
        return output
'''
##########DEFINITION DU MODELE AVEC 2 ENTREES ###########
'''
input1 = Input(shape=(50,))
embedding_layer = Embedding(len(word_index)+1,100,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False)(input1)
model1 = Bidirectional(LSTM(32))(embedding_layer)
model1 = Dense(64, activation='relu')(model1)


input2 = Input(shape=(224,224,3))
conv_layer = Conv2D(filters=12, kernel_size=5,kernel_constraint=CustomConstraint(custom_weights), padding='same', use_bias=False)(input2)
model = Conv2D(filters=16,kernel_size=3, padding='same',use_bias=False)(conv_layer)
model = BatchNormalization(axis=3, scale=False)(model)
model = Activation('relu')(model)
# Pooling layer
model = MaxPooling2D(pool_size=(4, 4),
                       strides=(4, 4),
                       padding='same')(model)

# Second convolution layer
model = Conv2D(filters=32,
                 kernel_size=3, 
                 padding='same',
                 use_bias=False)(model)
model = BatchNormalization(axis=3, scale=False)(model)
model = Activation('relu')(model)
model = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(model)
model = Dropout(0.2)(model)
# Third convolution layer
model = Conv2D(filters=64,
                 kernel_size=3, 
                 padding='same',
                 use_bias=False)(model)
model = BatchNormalization(axis=3, scale=False)(model)
model = Activation('relu')(model)
model = GlobalAveragePooling2D()(model)
# Fully connected layers
model = Dense(128, activation='relu')(model)

concat = layers.Concatenate()([model1,model])

final_model_output = Dense(2, activation='softmax')(concat)
final_model = Model(inputs=[input1,input2], outputs=final_model_output)
                    
# Compile the CNN Model
final_model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", f1_m])
final_model.summary()
'''
class CustomConstraint(Constraint):
    n=5
    
    def __init__(self,k,s):
        self.k = k
        self.s = s
        np.random.rand2 = lambda *args, dtype=np.float32: np.random.rand(*args).astype(dtype)
        f1 = np.random.rand2(s,s) #une matrice de nombres réeles aléatoires de dimension 5
        custom_weights = np.array((f1, f1, f1))
        f2 = custom_weights.transpose(1, 2, 0)
        self.custom_weights = tf.Variable(f2)
        #custom_weights = np.tile(f2, (k, 1, 1))
        #T2 = np.reshape(custom_weights,(k,s,s,3))
        #custom_weights = T2.transpose(1, 2, 3, 0)
        #self.custom_weights = tf.Variable(custom_weights)
    def __call__(self, weights):
        weights = self.custom_weights
        row_index = self.s//2
        col_index = self.s//2
        new_value = 0
        weights[row_index,col_index,:,:].assign(new_value)
        som = tf.keras.backend.sum(weights)
        sum_without_center1 = 1/som
        newMatrix = weights*sum_without_center1
        weights.assign(newMatrix)
        new_value = -1
        weights[row_index,col_index,:,:].assign(new_value)
        return weights

class CenterSumConstraint(Constraint):
    def __call__(self, weights):
        #value = weights.value().numpy()
        rows, cols = weights.shape
        #long = len(weights)
        weights[rows // 2, cols // 2]=0
        weights = weights / (weights_sum + 1e-8)  # Normalisation des poids
        weights=tf.tensor_scatter_nd_update(weights, [[rows // 2, cols // 2]], [-1])
        return weights
def fake_virtual():
    input1 = Input(shape=(100,))
    embedding_layer = Embedding(len(word_index)+1,100,embeddings_initializer=keras.initializers.Constant(embedding_matrix),input_length=100,trainable=False)(input1)
    model1 = Bidirectional(LSTM(32))(embedding_layer)
    out1 = Dense(64, activation='softmax')(model1)
    #final_model = tf.keras.Model(inputs=input1, outputs=out)
    #model = keras.Model(inputs=input1, outputs=out)
    #final_model_out = model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy","precision", f1_m])
    #return model
    input2 = Input(shape=(224,224,3))
    conv_layer = Conv2D(filters=3, kernel_size=5,kernel_constraint=CenterSumConstraint(), padding='same', use_bias=False)(input2)
    model = Conv2D(filters=16,kernel_size=3, padding='same',use_bias=False)(conv_layer)
    model = BatchNormalization(axis=3, scale=False)(model)
    model = Activation('relu')(model)
    model = Conv2D(filters=32,kernel_size=3, padding='same',use_bias=False)(model)
    model = BatchNormalization(axis=3, scale=False)(model)
    model = Activation('relu')(model)
    model = GlobalAveragePooling2D()(model)
    # Fully connected layers
    out2 = Dense(64, activation='softmax')(model)
    #concat = layers.Concatenate()([model1,model])
    outFinal = tf.keras.layers.Add()([out1, out2])
    final_model_output = Dense(1, activation='sigmoid')(outFinal)
    final_model = Model(inputs=[input1,input2], outputs=final_model_output)
    #final_model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy", f1_m])
    return final_model


#kf = KFold(n_splits = 5)
#skf = StratifiedKFold(n_splits = 5, shuffle = True) 
kfold = KFold(n_splits=5, shuffle=True)
print('DEBUT FORMATION DU MODEL........')
VALIDATION_ACCURACY = []
VALIDAITON_LOSS = []
save_dir = '/saved_models/'
fold_var = 1

for train_indices, val_indices in kfold.split(myTrain_Glove):
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    train_features1, train_features2 = myTrain_Glove[train_indices], imageListe[train_indices]
    val_features1, val_features2 = myTrain_Glove[val_indices], imageListe[val_indices]
    train_labels, val_labels = y[train_indices], y[val_indices]
    print(fold_var)
    model = fake_virtual()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+'model_'+str(fold_var)+'.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit([train_features1,train_features2],train_labels, epochs=10, batch_size=70,callbacks=callbacks_list, validation_data=([val_features1, val_features2],val_labels))
    model.load_weights("/saved_models/model_"+str(fold_var)+".h5")
    results = model.evaluate([val_features1, val_features2],val_labels,verbose=0)
    results = dict(zip(model.metrics_names,results))
    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
    y_pred = model.predict([myTrain_Glove[val_index],imageListe[val_index]]).ravel()
    tf.keras.backend.clear_session()
    
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(accuracy) + 1)
    plt.figure()
    plt.plot(epochs, accuracy, 'b', label='Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Val Accuracy',linestyle='-')
    plt.title('Accuracy et Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('courbes_Accuracy_'+str(fold_var)+'.png')
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'b', label='loss')
    plt.plot(epochs, val_loss, 'r', label='Val loss',linestyle='.')
    plt.title('Loss et Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('courbes_loss_'+str(fold_var)+'.png')
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend()
    plt.savefig('courbes_ROC_'+str(fold_var)+'.png')
    fold_var += 1
    

print(VALIDATION_ACCURACY)
print(VALIDATION_LOSS)
'''
print(len(labelText))
print(len(textListe))
modellstmP = fake_virtual()
modellstmP.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
modellstmP.fit([myTrain_Glove,imageListe], y,batch_size=60, epochs=10, verbose=1)

model = KerasClassifier(model=modellstmP, verbose=0)
batch_size = [5, 10]
epochs = [5, 10]
param_grid = dict(batch_size=batch_size, epochs=epochs)
#grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=3)
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=4,
    cv=5,
    refit=True,
    return_train_score=True
)
print("DEBUT GRIDSEARCH ...")
grid_result = grid.fit(myTrain_Glove, y)
print("####  summarize results1 ####")
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print("####  summarize results2 ####")
# Read the cv_results property into adataframe & print it out
cv_results_df = pd.DataFrame(grid_result.cv_results_)
print(cv_results_df)

# Extract and print the column with a dictionary of hyperparameters used
column = cv_results_df.loc[:, ["params"]]
print(column)

# Extract and print the row that had the best mean test score
best_row = cv_results_df[cv_results_df['rank_test_score'] == 1]
print(best_row)
'''
