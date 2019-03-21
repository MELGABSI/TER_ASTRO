#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
#import feets 
import astropy as ass
import itertools
import matplotlib.pyplot as plt
import pandas as pd
# lstm autoencoder recreate sequence
from keras.models import Sequential
from keras.layers import LSTM,Masking
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model


# In[ ]:


tab = pd.read_csv("Data/table.csv", sep=";")
tab.drop(columns = ['Unnamed: 0','None','object_id','passband_ne'], inplace=True)
table = tab
table[table != 0] = 1
tt = table.values
t = tab.values


# In[ ]:


# define input sequence
sequence = t
sequence_mask = tt
# reshape input into [samples, timesteps, features]
n_in = len(sequence)
n_msk = len(tt[0])
sequence = sequence.reshape((t.shape[0], n_in, 1))
sequence_mask = sequence_mask.reshape((tt.shape[0], n_msk, 1))


# In[ ]:


model = Sequential()
model.add(LSTM(1000, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_in))
model.add(Masking(mask_value = sequence_mask))
model.add(LSTM(1000, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')


# In[ ]:


# fit model
model.fit(sequence, sequence, epochs=100, verbose=0)
yhat = model.predict(sequence, verbose=0)


# In[ ]:


yhat_tab = pd.DataFrame(yhat)
yhat_tab.to_csv("y_pred.csv", sep=";",encoding="utf-8")

