import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Load the data
data = pd.read_csv('LCAlgarve2.csv')
data['DateTime'] = pd.to_datetime(data['DayCode1'])
data['daycode'] = data['DateTime'].dt.dayofweek.astype(str) + data['DateTime'].dt.hour.astype(str)

# print(data['daycode'])


# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
data[['Consumption']] = scaler_y.fit_transform(data[['Power1']])
data[['daycode']] = scaler_X.fit_transform(data[['daycode']])

# Prepare the data for training and testing
lookback = 96  # 24 hours * 4 (15-minute intervals per hour)
horizon = 8    # 2 days ahead * 4 (15-minute intervals per hour)
X = []
y = []
for i in range(len(data)-lookback-horizon+1):
    X.append(data[['daycode', 'Power1']].values[i:(i+lookback), :])
    y.append(data['Power1'].values[(i+lookback):(i+lookback+horizon)])
X = np.array(X)
y = np.array(y)

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
train_X, test_X = X[0:train_size,:,:], X[train_size:len(X),:,:]
train_y, test_y = y[0:train_size,:], y[train_size:len(y),:]

# Define the NARX model
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(horizon))
model.compile(loss='mse', optimizer='adam')

# Fit the NARX model to the training data
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# Make predictions on the test data
y_pred = model.predict(test_X)
y_pred = scaler_y.inverse_transform(y_pred)
test_y = scaler_y.inverse_transform(test_y)

# Plot the predicted vs. actual electricity consumption for the next 2 days with a 15-minute time interval
plt.figure(figsize=(20,6))
plt.plot(test_y[-48*2:, 0], label='Actual')
plt.plot(y_pred[-48*2:, 0], label='Predicted')
plt.title('Electricity Consumption Prediction for Next 2 Days')
plt.xlabel('Time (15-minute intervals)')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('NARX Model Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
