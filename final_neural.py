import pandas as pd
import scipy.stats as stats
import seaborn as sns
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

# Import warnings and set filter to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary functions from keras
# from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Import mean squared error and mean absolute error from sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


#Load dataset
#dataset 1
#data        = pd.read_csv("LCAlgarvetest.csv")

#dataset 2
data = pd.read_csv('LCAlgarve2.csv')


#Global dataset by array  variables
dayCode    = data["DayCode"]
pwr        = data['Power']
occ        = data["Occupation"]

# Align dataset

#Sort day code by ascending
# dayCodeSort = data.sort_values('DayCode1',ascending=True)
dayCodeSort =data.sort_values(by="DayCode", ascending= True)

# Print the number of rows and columns in the data
print('Number of rows and columns:', data.shape)
print(data.head(5))


# Get the information about the dataframe
print("\nInformation about the dataframe:")
print(data.info())

# Get the data type of each column in the dataframe
print("\nData type of each column in the dataframe:")
print(data.dtypes)

#Testing for Normality
#We will use D’Agostino’s K^2 Test to determine if our data is normally distributed.
#In the SciPy implementation of the test, the p-value will be used to make the following interpretation:
stat, p = stats.normaltest(data.Power)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# Set the significance level
alpha = 0.05

# Make a decision on the test result
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')

print( 'Kurtosis of normal distribution: {}'.format(stats.kurtosis(data.Power)))
print( 'Skewness of normal distribution: {}'.format(stats.skew(data.Power)))


# Calculate summary statistics by occupation
stats = data.groupby('DayCode').agg({'Occupation': ['mean', 'std']})
stats.columns = [' '.join(col).strip() for col in stats.columns.values]
# Create a normal plot using seaborn
sns.displot(data, x='Power', hue='Occupation', kind='kde', fill=True)
# Show the plot
# plt.show()

#Plot from Consumption for Dataset
# Exploratory Data Analysis(EDA)
plt.figure(figsize=(14,6))
plt.plot(data['Power'], color='purple')
plt.ylabel('Power', fontsize=12)
plt.xlabel('Date', fontsize=12)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
plt.tick_params(bottom = False)
plt.title('Power Consumption for Dataset', fontsize=14)
plt.tight_layout()
plt.grid(True)
sns.despine(bottom=True, left=True)
# plt.show()

# Plotting the histogram and normal probability plot for 'Power' column
plt.figure(figsize=(15,7))
# Histogram of 'Global_active_power' column
plt.subplot(1,2,1)
pwr.hist(bins=70, color='purple')
plt.title('Power Distribution', fontsize=16)

#TODO não consigo meter este grafico a funcionar
# # # Normal Probability Plot of 'Power' column
# plt.subplot(1,2,2)
# # Create the normal probability plot using stats.probplot
# stats.probplot(data, plot=plt, fit=True, rvalue=True)
# # Add a line to the plot
# plt.plot([0, max(pwr)], [0, max(pwr)], color='purple', linestyle='--')
# plt.title('Normal Probability Plot Power', fontsize=14)


# Printing the summary statistics of 'Global_active_power' column
df= data.describe().T
# print(df.to_string())
plt.show()

# #Modelling and Evaluation
# #Transform the Global_active_power column of the data DataFrame into a numpy array of float values
# Prepare data
data = data[['Occupation', 'Power']]
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
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(32))
model.add(Dense(forecast_steps))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1, shuffle=False)

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

model.summary()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')


# Plot the true and predicted values for the test set
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label='True')
plt.plot(predicted_values, label='Predicted')
plt.title('Electricity Consumption - True vs. Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.show()