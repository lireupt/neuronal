import pandas as pd
import scipy.stats as stats
import seaborn as sns
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import necessary functions from keras
# from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout

# Import mean squared error and mean absolute error from sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#Load dataset
#dataset 1
#data = pd.read_csv("LCAlgarvetest.csv")

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
plt.figure(figsize=(15,7))
stats = data.groupby('DayCode').agg({'Occupation': ['mean', 'std']})
stats.columns = [' '.join(col).strip() for col in stats.columns.values]
# Create a normal plot using seaborn
sns.displot(data, x='Power', hue='Occupation', kind='kde', fill=True)
# Show the plot
# plt.show()

# Plot the first subplot showing the violinplot of power
plt.figure(figsize=(15,7))
# Adjust the subplot's width
plt.subplots_adjust(wspace=0.2)
# Create the violinplot using Seaborn's violinplot function
sns.violinplot(x=dayCode, y=pwr, data=data, color='purple')
# Label the x-axis
plt.xlabel('DayCode', fontsize=12)
# Add a title to the plot
plt.title('Violin plot of Power', fontsize=14)
# Remove the top and right spines of the plot
sns.despine(left=True, bottom=True)
# Add a tight layout to the plot
plt.tight_layout()


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
# stats.probplot(pwr, plot=plt, fit=True, rvalue=True)
# # Add a line to the plot
# plt.plot([0, max(pwr)], [0, max(pwr)], color='purple', linestyle='--')
# plt.title('Normal Probability Plot Power', fontsize=14)

# Printing the summary statistics of 'Power' column
df= data.describe().T
# print(df.to_string())
# plt.show()

# #Modelling and Evaluation
# #Transform the Global_active_power column of the data DataFrame into a numpy array of float values
# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
data[['Power']] = scaler_y.fit_transform(data[['Power']])
data[['Occupation']] = scaler_X.fit_transform(data[['Occupation']])

# Prepare the data for training and testing
lookback = 96  # 24 hours * 4 (15-minute intervals per hour)
horizon = 8    # 2 days ahead * 4 (15-minute intervals per hour)

X = []
y = []
for i in range(len(data)-lookback-horizon+1):
    X.append(data[['Occupation', 'Power']].values[i:(i+lookback), :])
    y.append(data['Power'].values[(i+lookback):(i+lookback+horizon)])
X = np.array(X)
y = np.array(y)

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
train_X, test_X = X[0:train_size,:,:], X[train_size:len(X),:,:]
train_y, test_y = y[0:train_size,:], y[train_size:len(y),:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(horizon))
model.compile(loss='mse', optimizer='adam')

# Fit the LSTM model to the training data
history = model.fit(train_X, train_y, epochs=5, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# Displaying a summary of the model
model.summary()

# Make predictions on the test data
y_pred = model.predict(test_X)
y_pred = scaler_y.inverse_transform(y_pred)
test_y = scaler_y.inverse_transform(test_y)


# # Evaluate the model
train_score = model.evaluate(train_X, train_y, verbose=0)
test_score = model.evaluate(test_X, test_y, verbose=0)
print('Train Score: {:.2f} MSE ({:.2f} RMSE)'.format(train_score, np.sqrt(train_score)))
print('Test Score: {:.2f} MSE ({:.2f} RMSE)'.format(test_score, np.sqrt(test_score)))


# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Plot the predicted vs. actual electricity consumption for the next 2 days with a 15-minute time interval
plt.figure(figsize=(20,6))
plt.plot(test_y[-48*2:, 0], label='Actual')
plt.plot(y_pred[-48*2:, 0], label='Predicted')
plt.title('Electricity Consumption Prediction for Next 2 Days')
plt.xlabel('Time (15-minute intervals)')
plt.ylabel('Electricity Consumption')
plt.legend()

plt.show()
