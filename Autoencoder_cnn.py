#!/usr/bin/env python
# coding: utf-8

# In[1]:


# lstm autoencoder predict sequence
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Input, Conv1D, MaxPooling1D, Flatten,UpSampling1D
from keras.layers import Dense, Reshape, Masking
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from keras.models import Model
from sklearn import preprocessing
import keras.backend as K
import keras.callbacks


# In[3]:


tab = pd.read_csv("Data/table.csv", sep=";")


# In[4]:

tab.drop(columns = ['Unnamed: 0','None','object_id','passband_ne'], inplace=True)
t = tab.values
min_max_scaler = preprocessing.MinMaxScaler()
t = min_max_scaler.fit_transform(t)
mask = tab
mask[mask!=0]=1
mask = mask.values

s = mask * t 

# In[11]:


x_train = s.reshape((len(s), 1095, 1))
y_train = s.reshape((len(s), 1095, 1))


# In[8]:


input_sig = Input(batch_shape=(72,1095,1))
out = TimeDistributed(Masking(mask_value=0))(input_sig)
x0 = Conv1D(512,5, activation='relu', padding='valid')(out)
x1_ = MaxPooling1D(2)(x0)
x = Conv1D(128,5, activation='relu', padding='valid')(x1_)
x1 = MaxPooling1D(2)(x)
x2 = Conv1D(32,5, activation='relu', padding='valid')(x1)
x3 = MaxPooling1D(2)(x2)
flat = Flatten()(x3)
encoded = Dense(32,activation = 'relu')(flat)

print("shape of encoded {}".format(K.int_shape(flat)))

x2_ = Conv1D(32, 5, activation='relu', padding='valid')(x3)
x1_ = UpSampling1D(2)(x2_)
x_ = Conv1D(128, 5, activation='relu', padding='valid')(x1_)
upsamp = UpSampling1D(2)(x_)
x_0 = Conv1D(512,5, activation='relu', padding='valid')(upsamp)
flat = Flatten()(x_0)
decoded = Dense(1095,activation = 'relu')(flat)
decoded = Reshape((1095,1))(decoded)

print("shape of decoded {}".format(K.int_shape(x1_)))

autoencoder = Model(input_sig, decoded)
autoencoder.compile(optimizer='adam', loss='mse')


# In[9]:


autoencoder.summary()
encoder = Model(input_sig, encoded)
features = encoder.predict(x_train, batch_size=72)
np.save("Data/feat.npy", features)

# In[10]:


autoencoder.fit(x_train,y_train,
                      batch_size=72,
                      epochs=3,
                      verbose=1)

