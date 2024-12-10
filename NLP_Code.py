import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import time
df = pd.read_csv("C:/Users/USER/Downloads/archive/twitter_training.csv", names=['ColumnA','ColumnB','Target','Sentence'])
df2 = pd.read_csv("C:/Users/USER/Downloads/archive/twitter_validation.csv", names=['ColumnA','ColumnB','Target','Sentence'])
df = df.drop(df[['ColumnA','ColumnB']], axis=1)
df2 = df2.drop(df2[['ColumnA','ColumnB']], axis=1)
df = df[['Sentence','Target']]
df2 = df2[['Sentence','Target']]
df = df[df['Target'] != 'Irrelevant'] 
df2 = df2[df2['Target'] != 'Irrelevant']
df['Target'] = df['Target'].replace({'Positive':2,'Neutral':1,'Negative':0})
df2['Target'] = df2['Target'].replace({'Positive':2,'Neutral':1,'Negative':0})

import string
def remove_punct(text):
    if not isinstance(text, str):
        return str(text)
    translator = str.maketrans("","",string.punctuation)
    return text.translate(translator)
df['Sentence'] = df.Sentence.map(remove_punct) 
df2['Sentence'] = df2.Sentence.map(remove_punct) 

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
def remove_stopwrds(text):
    if not isinstance(text, str):
        return str(text)
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return ' '.join(filtered_words)
df['Sentence'] = df.Sentence.map(remove_stopwrds) 
df2['Sentence'] = df2.Sentence.map(remove_stopwrds) 

train_sentences = df.Sentence.to_numpy()
train_labels = df.Target.to_numpy()
test_sentences = df2.Sentence.to_numpy()
test_labels = df2.Target.to_numpy()
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words= 20000)
tokenizer.fit_on_texts(df['Sentence'])
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

from tensorflow.keras.preprocessing.sequence import pad_sequences
train_padded = pad_sequences(train_sequences, maxlen=20,padding='post',truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=20,padding='post',truncating='post')
from tensorflow.keras import layers
model = keras.models.Sequential()
model.add(layers.Embedding(20000,32,input_length=20))
model.add(layers.LSTM(64,dropout=0.1))
model.add(layers.Dense(3,activation='softmax'))

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']
model.compile(loss=loss,optimizer=optim,metrics=metrics)
model.fit(train_padded,train_labels,epochs=20,validation_data=(test_padded,test_labels), verbose=2)

prediction = model.predict(train_padded)
predicted_classes = prediction.argmax(axis=-1)
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
predicted_labels = [label_mapping[cls] for cls in predicted_classes]
print(test_sentences[10:30])
print(test_labels[10:30])
print(prediction[10:30])