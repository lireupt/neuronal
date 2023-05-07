# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
df = pd.read_csv('LCAlgarve1.csv', parse_dates=True, index_col='DayCode1')

# Prepare data
data = df[['Occupation1', 'Power1']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Define time steps and forecast steps
time_steps = 96 # 15 minutes per time step
forecast_steps = 48 # 12 hours ahead

# Create input and output sequences
X = []
y = []
for i in range(time_steps, len(scaled_data) - forecast_steps):
    X.append(scaled_data[i-time_steps:i])
    y.append(scaled_data[i:i+forecast_steps, 1]) # select only power variable for y

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(32))
model.add(Dense(forecast_steps))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=1, shuffle=False)

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

# Plot the true and predicted values for the test set
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label='True')
plt.plot(predicted_values, label='Predicted')
plt.title('Electricity Consumption - True vs. Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Electricity Consumption (kWh)')
plt.legend()
plt.show()