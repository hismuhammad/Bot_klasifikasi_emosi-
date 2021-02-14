#!/usr/bin/env python
# coding: utf-8

# # Klasifikasi Emosi Pada Teks Menggunakan RNN <a class="anchor" id="titlepage"></a>

# <img src="future.jpg" />

# In[1]:

import os
import preprocessor as p
import numpy as np 
import pandas as pd 
import emoji
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from tqdm import tqdm
from keras.models import model_from_json


# In[2]:
modulePath = os.path.dirname(__file__)
filePath = os.path.join(modulePath, 'bankdata.xlsx')
data = pd.read_excel(filePath)


# In[3]:

typoFile = os.path.join(modulePath, 'typo.txt')
misspell_data = pd.read_csv(typoFile,sep=":",names=["correction","misspell"])
misspell_data.misspell = misspell_data.misspell.str.strip()
misspell_data.misspell = misspell_data.misspell.str.split(" ")
misspell_data = misspell_data.explode("misspell").reset_index(drop=True)
misspell_data.drop_duplicates("misspell",inplace=True)
miss_corr = dict(zip(misspell_data.misspell, misspell_data.correction))

#Sampel isi kamus
{v:miss_corr[v] for v in [list(miss_corr.keys())[k] for k in range(5)]}


# In[4]:


def misspelled_correction(val):
    for x in val.split(): 
        if x in miss_corr.keys(): 
            val = val.replace(x, miss_corr[x]) 
    return val

data["clean_content"] = data.content.apply(lambda x : misspelled_correction(x))


# In[5]:


p.set_options(p.OPT.MENTION, p.OPT.URL)
p.clean("Mau kemana guys @alx #sport🔥 12458776")


# In[6]:


data["clean_content"]=data.content.apply(lambda x : p.clean(x))


# In[7]:


def punctuation(val): 
  
    punctuations = '''()-[]{};:'"\,<>./@#$%^&_~'''
  
    for x in val.lower(): 
        if x in punctuations: 
            val = val.replace(x, " ") 
    return val


# In[8]:


punctuation("test ombak@ #ldfldlf??? !! ")


# In[9]:


data.clean_content = data.clean_content.apply(lambda x : ' '.join(punctuation(emoji.demojize(x)).split()))


# In[10]:


def clean_text(val):
    val = misspelled_correction(val)
    val = p.clean(val)
    val = ' '.join(punctuation(emoji.demojize(val)).split())
    
    return val


# In[11]:


clean_text("saya punya ide💡 bag00ss@@ ! ? ")


# In[12]:


data = data[data.clean_content != ""]


# In[13]:


data.emotion.value_counts()


# In[14]:


sent_to_id  = {"kegembiraan":1,"kesedihan":2,"malu":3,"marah":4,
                        "menjijikkan":5,"takut":6,"kesalahan":7}


# In[15]:


data["emotion_id"] = data['emotion'].map(sent_to_id)


# In[16]:


data


# In[17]:


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data.emotion_id)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(data.clean_content,Y, random_state=1995, test_size=0.2, shuffle=True)


# In[19]:


# menggunakan tokenizer dari Keras
token = text.Tokenizer(num_words=None)
max_len = 160
Epoch = 5
token.fit_on_texts(list(X_train) + list(X_test))
X_train_pad = sequence.pad_sequences(token.texts_to_sequences(X_train), maxlen=max_len)
X_test_pad = sequence.pad_sequences(token.texts_to_sequences(X_test), maxlen=max_len)


# In[20]:


w_idx = token.word_index


# In[21]:

load_json = os.path.join(modulePath, 'model.json')


json_file = open(load_json, "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
load_h5= os.path.join(modulePath,'model.h5')
loaded_model.load_weights(load_h5)


# In[22]:


def get_emotion(model,text):
    text = clean_text(text)
    #tokenize
    words = token.texts_to_sequences([text])
    words = sequence.pad_sequences(words, maxlen=max_len, dtype='int64')
    emotion = model.predict(words,batch_size=1,verbose = 2)
    sent = np.round(np.dot(emotion,100).tolist(),0)[0]
    result = pd.DataFrame([sent_to_id.keys(),sent]).T
    result.columns = ["emotion","percentage"]
    result=result[result.percentage !=0]
    return result


# In[ ]:





# In[ ]:


def plot_result(df):
    #colors=['#D50000','#000000','#008EF8','#F5B27B','#EDECEC','#D84A09','#019BBD','#FFD000','#7800A0','#098F45','#807C7C','#85DDE9','#F55E10']
    #fig = go.Figure(data=[go.Pie(labels=df.sentiment,values=df.percentage, hole=.3,textinfo='percent',hoverinfo='percent+label',marker=dict(colors=colors, line=dict(color='#000000', width=2)))])
    #fig.show()
    colors={'kegembiraan':'rgb(9,143,69)',
                    'kesedihan':'rgb(64,64,64)','marah':'rgb(204,0,0)',
                    'menjijikkan':'rgb(153,204,0)','malu':'rgb(255,153,153)',
                    'kesalahan':'rgb(122,0,204)',
                    'takut':'rgb(255,153,0)'}
    col_2={}
    for i in result.emotion.to_list():
        col_2[i]=colors[i]
    #fig = px.pie(df, values='percentage', names='emotion',color='emotion',color_discrete_map=col_2,hole=0.3)
    #fig.show()


# In[23]:


def get_emotionn(model,text):
    text = clean_text(text)
    #tokenize
    words = token.texts_to_sequences([text])
    words = sequence.pad_sequences(words, maxlen=max_len, dtype='int64')
    emotion = model.predict(words,batch_size=1,verbose = 2)
    sent = np.round(np.dot(emotion,100).tolist(),0)[0]
    result = pd.DataFrame([sent_to_id.keys(),sent]).T
    result.columns = ["emotion","percentage"]
    result=result[result.percentage !=0]
    best_result = result.sort_values('percentage',ascending=False).head(1)
    the_result = best_result.iloc[0]['emotion']
    return "Emosi yang dirasakan " + the_result


# In[ ]:


result = get_emotion(loaded_model,"Teman saya dipukul oleh orang tidak dikenal")
plot_result(result)
result = get_emotion(loaded_model,"Dia berbohong tentang pekerjaannya")
plot_result(result)
result = get_emotion(loaded_model,"Di ulang tahun ke 23 saya diberi hadiah motor oleh sahabat terbaik saya")
plot_result(result)


# In[24]:


hasil =get_emotionn(loaded_model,"Kamu hitam tapi manis")
print(hasil)
hasil =get_emotionn(loaded_model,"Pamanku datang dari desa membawa hadiah")
print(hasil)


# <a href="#titlepage"><img  src="https://za.heytv.org/wp-content/uploads/2019/08/AGF-l79DYZtk_pSyfWgIP3D-3yi8YN6ZeWO0E8tyLgs800-c-k-c0xffffffff-no-rj-mo.jpeg" style="height: 300px"/></a>
