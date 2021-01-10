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
#%%
from numpy import hstack

# split a multivariate sequence into samples
# x component has a three-dimensional structure
def split_multi_seq(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

#%%
""""
Multiple Input Series

eg
input:
10, 15
20, 25
30, 35

output:
65
""" 
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# multivariate data preparation
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1)) # output series

# horizontally stack columns
# each row is a time step, and each column is a separate time series
# standard way of storing parallel time series in a CSV file
dataset = hstack((in_seq1, in_seq2, out_seq))

# number of time steps
n_steps = 3
# convert into input/output
x, y = split_multi_seq(dataset, n_steps)

# x.shape = (number of samples, number of time steps per sample, number of parallel time series)
print(x.shape, y.shape)

# summarize the data (2 input series, output)
for i in range(len(x)):
	print(x[i], y[i])

#%%
"""
Vanilla LSTM

"""
n_steps = 3
n_features = x.shape[2]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x, y, epochs=200, verbose=0)

x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))

# prediction
yhat = model.predict(x_input, verbose=0)
print(yhat)

#%%
"""
Multiple Parallel Series/ multivariate forecasting

input:
10, 15, 25
20, 25, 45
30, 35, 65

output:
40, 45, 85
"""

# split a multivariate sequence into samples
# X is three-dimensiona = number of sample, number of time steps chosen per sample, number of parallel time series or features
# y is two-dimensional = number of samples, number of time variables per sample to be predicted
def split_parallel_seq(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

# choose a number of time steps
n_steps = 3

# convert into input/output
x, y = split_parallel_seq(dataset, n_steps)

print(x.shape, y.shape)

# summarize the data
for i in range(len(x)):
	print(x[i], y[i])
    
#%%
"""
Stacked LSTM
"""

n_steps = 3
n_features = x.shape[2] # dataset knows the number of features


# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(x, y, epochs=400, verbose=0)
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#%%
"""
Univariate
Multi-Step LSTM Models

1. Vector Output Model
2. Encoder-Decoder Model
    
"""

"""
Data preprocessing
"""

def split_multistep_seq(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

#%%
"""
Vector Output Model

one time step of each output time series was forecasted as a vector
"""
# Stacked LSTM
# input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

# choose a number of time steps
n_steps_in, n_steps_out = 3, 2 # input, output steps 

# split into samples
x, y = split_multistep_seq(raw_seq, n_steps_in, n_steps_out)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
x = x.reshape((x.shape[0], x.shape[1], n_features))

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x, y, epochs=50, verbose=0)

# prediction
x_input = array([110, 120, 130])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#%%
"""
Encoder-Decoder Model

sequence-to-sequence / input and output sequences / seq2seq
1. Encoder = responsible for reading and interpreting the input sequence
    Traditionally a Vanilla LSTM model
    output = a fixed length vector that represents the model’s interpretation of the sequence
2. fixed-length output of the encoder is repeated, once for each required time step in the output sequence
3. Decoder =   output a value for each value in the output time step
"""
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# time steps
n_steps_in, n_steps_out = 3, 2

# split into samples
X, y = split_multistep_seq(raw_seq, n_steps_in, n_steps_out)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# output/y of the training dataset must be three-dimensional shape of [samples, timesteps, features]
y = y.reshape((y.shape[0], y.shape[1], n_features))

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
# use the same output layer or layers to make each one-step prediction in the output sequence.
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=100, verbose=0)

# prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#%%
"""
Multivariate Multi-Step LSTM Models

1. Multiple Input Multi-Step Output
2. Multiple Parallel Input and Multi-Step Output

"""

#%%
"""
1. Multiple Input Multi-Step Output

Stacked LSTM
Input:
10, 15
20, 25
30, 35

Output:
65
85
"""

# split a multivariate sequence into samples
def split_multi_variate_step(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

# time steps
n_steps_in, n_steps_out = 3, 2

# covert into input/output
x, y = split_multi_variate_step(dataset, n_steps_in, n_steps_out)

# the dataset knows the number of features, e.g. 2
n_features = x.shape[2]

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x, y, epochs=200, verbose=0)

# prediction
x_input = array([[70, 75], [80, 85], [90, 95]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#%%
"""
2. Multiple Parallel Input and Multi-Step Output

Input:
10, 15, 25
20, 25, 45
30, 35, 65

Output:
40, 45, 85
50, 55, 105
"""

# split a multivariate sequence into samples
def split_prl_multi_variate_step(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

# choose a number of time steps
n_steps_in, n_steps_out = 3, 2

# covert into input/output
x, y = split_prl_multi_variate_step(dataset, n_steps_in, n_steps_out)

# the dataset knows the number of features, e.g. 2
n_features = x.shape[2]

# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x, y, epochs=300, verbose=0)

# prediction
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)