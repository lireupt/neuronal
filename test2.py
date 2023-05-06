import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM


import matplotlib.pyplot as plt

import scipy.stats as stats
import seaborn as sns
import sys
np.set_printoptions(threshold=sys.maxsize)



# Import warnings and set filter to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary functions from keras
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers,models

# Import early stopping from keras callbacks
from keras.callbacks import EarlyStopping

# Import mean squared error and mean absolute error from sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Load the data
data = pd.read_csv("LCAlgarvetest.csv")

datapwr = data['Power1'].values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))

data['daycode_occupation'] = pd.Categorical(data['DayCode1'].astype(str) + '_' + data['Occupation1'].astype(str))
data['scaled_power'] = scaler.fit_transform(datapwr.reshape(-1,1))
data['scaled_daycode_occupation'] = scaler.fit_transform(data['daycode_occupation'].cat.codes.values.reshape(-1,1))

# print(data['scaled_power'])

# max_value = np.max(data['scaled_power'])
# print('Maximum value in array\n',max_value)

# Create sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length, :-1])
        y.append(data[i+seq_length, -1])
    return np.array(X), np.array(y)



def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X).astype(np.float32), np.array(Y).astype(np.float32)





seq_length = 24 # Length of input sequence
data = data.values
train_size = int(len(data) * 0.8)
train_data = data[:train_size,:]
test_data = data[train_size-seq_length:,:]

X_train, Y_train = create_dataset(train_data, seq_length)
X_test, Y_test = create_dataset(test_data, seq_length)


X_train.shape


Y_train.shape

# print(Y_train)

# reshape input to be [samples, time steps, features]
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

X_train.shape
print(X_train.shape)



# Build the LSTM model
model = models.Sequential()
model.add(layers.LSTM(50, input_shape=(seq_length, 2)))
model.add(layers.Dense(1))



# Train the model
# Defining the LSTM model
model = models.Sequential()
model.compile(optimizer='adam', loss='mse')
# model.compile(loss='mean_squared_error', optimizer='adam')

# Adding the first layer with 100 LSTM units and input shape of the data
model.add(layers.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))


history = model.fit(X_train, Y_train, epochs=20, batch_size=1240, validation_data=(X_test, Y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=4)], verbose=1, shuffle=False)

model.summary()
#

# history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))

# print(history)
# Evaluate the model
train_score = model.evaluate(X_train, Y_train, verbose=0)
test_score = model.evaluate(X_test, Y_test, verbose=0)
print('Train Score: {:.2f} MSE ({:.2f} RMSE)'.format(train_score, np.sqrt(train_score)))
print('Test Score: {:.2f} MSE ({:.2f} RMSE)'.format(test_score, np.sqrt(test_score)))

# Plot loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Predict on test set
Y_pred = model.predict(X_test)



# Inverse scaling of test data and predictions
# y_test_inv = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], Y_test.reshape(-1,1)), axis=1))[:, -1]
# y_pred_inv = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], Y_pred.reshape(-1,1)), axis=1))[:, -1]

Y_test_reshaped = Y_test.reshape((250, -1))
Y_pred_reshaped = Y_pred.reshape((250, -1))

y_test_inv = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], Y_test_reshaped), axis=1))[:, -1]
y_pred_inv = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], Y_pred_reshaped), axis=1))[:, -1]



# Plot actual vs predicted values
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('Actual vs Predicted Values')
plt.ylabel('Power')
plt.xlabel('Time')
plt.legend()
plt.show()
