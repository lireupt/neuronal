import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#
# # Create a sample data frame with random energy consumption and occupation data
# # data = {'Energy Consumption (kWh)': [20, 30, 25, 35, 40, 50, 45, 55],
# #         'Occupation': ['Engineer', 'Teacher', 'Engineer', 'Lawyer', 'Doctor', 'Lawyer', 'Doctor', 'Teacher']}
#
# df = pd.read_csv('LCAlgarve2.csv', parse_dates=True, index_col='DayCode1')
# df = pd.DataFrame(df)
#
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt


# Load the data

# df = pd.read_csv('LCAlgarve2.csv', parse_dates=True, index_col='DayCode1')
# df = pd.DataFrame(df)


data = pd.read_csv('LCAlgarve2.csv', parse_dates=True, index_col='DayCode1')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values



# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
train_X, test_X = X[0:train_size,:], X[train_size:len(X),:]
train_y, test_y = y[0:train_size], y[train_size:len(y)]
print(train_X, test_X)
print(train_y, test_y)

#
# Define the NARX model
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[0], train_X.shape[1])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the NARX model to the training data
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# Evaluate the model
score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss:', score)

# Make predictions on the test data
y_pred = model.predict(test_X)

# Make predictions on the test data
y_pred = model.predict(test_X)

# Plot the predicted vs. actual electricity consumption
plt.plot(test_y, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Electricity Consumption Prediction')
plt.xlabel('Time')
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

