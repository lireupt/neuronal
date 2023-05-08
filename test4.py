# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Import mean squared error and mean absolute error from sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Load data
df = pd.read_csv('LCAlgarve2.csv')

# Prepare data
data = df[['Occupation', 'Power']]


data = data.values.astype('float32')
# print(data)
data = np.reshape(data, (-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1),copy=True, clip=False)


# scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

print(scaled_data)



# Define time steps and forecast steps
time_steps = 96 # 15 minutes per time step
forecast_steps = 48 # 12 hours ahead

# Create input and output sequences
X = []
y = []
for i in range(time_steps, len(scaled_data) - forecast_steps):
    X.append(scaled_data[i-time_steps:i])
    y.append(scaled_data[i:i+forecast_steps, 0]) # select only power variable for y

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


X_train.shape
print(X_train.shape)


# Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(32))
model.add(Dense(forecast_steps))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
history = model.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test), verbose=1, shuffle=False)

# Make predictions
test_data = X[-1] # select last sequence as test data
test_data = np.expand_dims(test_data, axis=0)
predictions = model.predict(test_data)

# Invert scaling and select only power variable
actual_values = scaler.inverse_transform(np.concatenate((np.zeros((forecast_steps, 1)), y[-1].reshape(-1, 1)), axis=1))[:, 1]
predicted_values = scaler.inverse_transform(np.concatenate((np.zeros((forecast_steps, 1)), predictions[0].reshape(-1, 1)), axis=1))[:, 1]

# Print results
print('Actual:', actual_values)
print('Predicted:', predicted_values)

import matplotlib.pyplot as plt




#Reshape the numpy array into a 2D array with 1 column



# # # invert predictions
# train_predict = scaler.inverse_transform(actual_values)
# Y_train = scaler.inverse_transform([y_train])
# test_predict = scaler.inverse_transform(predicted_values)
# Y_test = scaler.inverse_transform([y_test])
#
# print('Train Mean Absolute Error:', mean_absolute_error(y_train[0], train_predict[:,0]))
# print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(y_train[0], train_predict[:,0])))
# print('Test Mean Absolute Error:', mean_absolute_error(y_test[0], test_predict[:,0]))
# print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[0], test_predict[:,0])))
# #
#
# plt.figure(figsize=(8,4))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Test Loss')
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')
# # plt.show();
#
# # Plot the true and predicted values for the test set
# plt.figure(figsize=(12, 6))
# plt.plot(actual_values, label='True')
# plt.plot(predicted_values, label='Predicted')
# plt.title('Electricity Consumption - True vs. Predicted')
# plt.xlabel('Time Steps')
# plt.ylabel('Electricity Consumption (kWh)')
# plt.legend()
# plt.show()