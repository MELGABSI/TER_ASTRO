#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import LSTM,Masking
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import pandas as pd 
import numpy as np
from sklearn import preprocessing


# In[2]:


tab = pd.read_csv("Data/table.csv", sep=";")


# In[22]:


tab.drop(columns = ['Unnamed: 0','None','object_id','passband_ne'], inplace=True)


# In[25]:


t = tab.values
min_max_scaler = preprocessing.StandardScaler()
t = min_max_scaler.fit_transform(t)
#df_normalized = pd.DataFrame(np_scaled)


# In[33]:


# lstm autoencoder predict sequence
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
# define input sequence
seq_in = t[0]
# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))
# prepare output sequence
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
#plot_model(model, show_shapes=True, to_file='predict_lstm_autoencoder.png')
# fit model


# In[ ]:


for i in range(0,10):
    seq_in = t[i].reshape((1,len(t[i]), 1))
    model.fit(seq_in, seq_out, epochs=300, verbose=1)
    yhat = model.predict(seq_in, verbose=0)
    print(yhat[0,:,0])

