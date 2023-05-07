# Import necessary libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load data into a Pandas dataframe
data = pd.read_csv("LCAlgarve1.csv")

# Preprocess the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Define the number of previous time steps to use for predicting the next time step
time_steps = 96

# Define the number of steps ahead to forecast
forecast_steps = 48

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Define the input and output variables for the LSTM network
X_train, y_train = [], []
for i in range(time_steps, len(train_data)-forecast_steps):
    X_train.append(train_data[i - time_steps:i, :])
    y_train.append(train_data[i:i+forecast_steps, 2])
X_train, y_train = np.array(X_train), np.array(y_train)

X_test, y_test = [], []
for i in range(time_steps, len(test_data)-forecast_steps):
    X_test.append(test_data[i - time_steps:i, :])
    y_test.append(test_data[i:i+forecast_steps, 2])
X_test, y_test = np.array(X_test), np.array(y_test)

# Define the LSTM model
model = Sequential()


model.add(LSTM(64, activation='relu', input_shape=(time_steps, 3)))
model.add(Dense(forecast_steps))
model.compile(optimizer='adam', loss='mse')

# Train the LSTM model
model.fit(X_train, y_train, epochs=100, batch_size=64)

# Evaluate the LSTM model
mse = model.evaluate(X_test, y_test)
print('MSE: %.3f' % mse)

# Make predictions with the LSTM model
predictions = model.predict(X_test)

# Inverse scale the predictions and actual values to their original form
predictions = scaler.inverse_transform(np.concatenate((np.zeros((time_steps+forecast_steps, 2)), predictions), axis=1))[:, 2][time_steps:]
actual_values = scaler.inverse_transform(np.concatenate((np.zeros((time_steps+forecast_steps, 2)), test_data), axis=1))[:, 2][time_steps:-forecast_steps]

# Plot the predictions against the actual values
import matplotlib.pyplot as plt
plt.plot(actual_values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()