# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 19:31:12 2020

@author: Yu Zhe
"""

"""
Data preprocessing
"""
from numpy import array

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y) # x = [samples, timesteps]

#%%
"""
Demo
"""
# Splits the univariate series into six samples where each sample has three input time steps and one output time step.
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90] # Training dataset

# choose a number of time steps
n_steps = 3

# split into samples
x, y = split_sequence(raw_seq, n_steps)

# summarize the data
for i in range(len(x)):
	print(x[i], y[i])

n_features = 1 # univariate
# reshape from [samples, timesteps] into [samples, timesteps, features]
x = x.reshape((x.shape[0], x.shape[1], n_features))

#%%    
"""
Vanilla LSTM

Univariate time series forecasting
Contains single hidden layer of LSTM units, and an output layer used to make a prediction.
Three-dimensiona model input = [samples, timesteps, features]
"""
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

n_features = 1 # univariate

# define model 
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features))) # 50 LSTM units in the hidden layer
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse') #  Adam version of stochastic gradient descent 

# fit model
model.fit(x, y, epochs=200, verbose=0)

# Prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#%%
"""
Stacked LSTM

Multiple hidden LSTM layers can be stacked one on top of another.
"""
n_steps = 3
n_features = 1 # univariate

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features))) #3D output from hidden LSTM layer as input to the next.
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x, y, epochs=200, verbose=0)

# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#%%
"""
Bidirectional LSTM

LSTM model to learn the input sequence both forward and backwards and concatenate both interpretations.
Wrap the first hidden layer in a wrapper layer called Bidirectional.
"""
from keras.layers import Bidirectional

n_steps = 3
n_features = 1

# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x, y, epochs=200, verbose=0)

# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#%%
"""
CNN LSTM

CNN is used to interpret subsequences of input that together are provided as a sequence to an LSTM model to interpret
"""
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

n_steps = 4
# split into samples
x, y = split_sequence(raw_seq, n_steps)

# Each sample can then be split into two sub-samples, each with two time steps. 
# So the CNN can interpret each subsequence of two time steps and provide a 
# time series of interpretations of the subsequences to the LSTM model to process as input.
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
x = x.reshape((x.shape[0], n_seq, n_steps, n_features))

# define model
model = Sequential()
# TimeDistributed wrapper to apply the entire model once per input subsequence 
# Reuse the same CNN model when reading in each sub-sequence of data separately
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2))) # distills the filter maps down to 1/2 of their size that includes the most salient features
model.add(TimeDistributed(Flatten())) # single input time step
model.add(LSTM(50, activation='relu')) 
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x, y, epochs=500, verbose=0)

# demonstrate prediction
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#%%
"""
ConvLSTM

Convolutional reading of input is built directly into each LSTM unit
Developed for reading two-dimensional spatial-temporal data, 
but can be adapted for use with univariate time series forecasting
"""
from keras.layers import ConvLSTM2D


# choose a number of time steps
n_steps = 4

# split into samples
x, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
# as the layer expects input as a sequence of two-dimensional images
n_features = 1
n_seq = 2
n_steps = 2
# timesteps = number of subsequences, or n_seq
# columns = number of time steps for each subsequence, or n_steps
# number of rows is fixed at 1 for one-dimensional data
x = x.reshape((x.shape[0], n_seq, 1, n_steps, n_features))

# define model
model = Sequential()
# two-dimensional kernel size in terms of (rows, columns)
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x, y, epochs=500, verbose=0)

# demonstrate prediction
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#%%
"""
Multivariate LSTM Models

1. Multiple Input Series
2. Multiple Parallel Series
"""

# Multiple Input Series





