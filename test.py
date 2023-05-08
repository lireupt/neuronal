import pandas as pd
import scipy.stats as stats
import seaborn as sns
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# Import warnings and set filter to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary functions from keras
# from tensorflow import keras
from tensorflow.keras import layers,models

# Import early stopping from keras callbacks
from keras.callbacks import EarlyStopping

# Import mean squared error and mean absolute error from sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


#Load dataset
#dataset 1
#data        = pd.read_csv("LCAlgarvetest.csv")

#dataset 2
data = pd.read_csv("LCAlgarve2.csv")

#Global dataset by array  variables
dayCode1    = data["DayCode1"]
pwr1        = data['Power1']
occ1        = data["Occupation1"]

#Sort day code by ascending
# dayCodeSort = data.sort_values('DayCode1',ascending=True)
dayCodeSort =data.sort_values(by="DayCode1", ascending= True)
# print(dayCodeSort)

#Align dataset
# Print the number of rows and columns in the data
print('Number of rows and columns:', data.shape)
print(data.head(5))

# Get the information about the dataframe
print("\nInformation about the dataframe:")
print(data.info())

# Get the data type of each column in the dataframe
print("\nData type of each column in the dataframe:")
print(data.dtypes)


#Convert data type to float in columns Occupation2
# data['Occupation1'] = pd.to_numeric(data['Occupation1'], errors='coerce', downcast="integer")
# data['Occupation2'] = pd.to_numeric(data['Occupation2'], errors='coerce', downcast='float')
# print(data.dtypes)
# print(data)


#Testing for Normality
#We will use D’Agostino’s K^2 Test to determine if our data is normally distributed.
#In the SciPy implementation of the test, the p-value will be used to make the following interpretation:
stat, p = stats.normaltest(data.Power1)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# Set the significance level
alpha = 0.05

# Make a decision on the test result
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')

print( 'Kurtosis of normal distribution: {}'.format(stats.kurtosis(data.Power1)))
print( 'Skewness of normal distribution: {}'.format(stats.skew(data.Power1)))


# Calculate summary statistics by occupation
stats = data.groupby('DayCode1').agg({'Occupation1': ['mean', 'std']})
stats.columns = [' '.join(col).strip() for col in stats.columns.values]
# Create a normal plot using seaborn
sns.displot(data, x='Power1', hue='Occupation1', kind='kde', fill=True)
# Show the plot
# plt.show()

#TODO ver erro da biblioteca do STATS
# #Plot from Consumption for Dataset
# # Exploratory Data Analysis(EDA)
# plt.figure(figsize=(14,6))
# # plt.plot(data['Power1'], color='purple')
# plt.ylabel('Power', fontsize=12)
# plt.xlabel('Date', fontsize=12)
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# plt.tick_params(bottom = False)
# plt.title('Power Consumption for Dataset', fontsize=14)
# plt.tight_layout()
# plt.grid(True)
# sns.despine(bottom=True, left=True)
# # plt.show()


# Plot the first subplot showing the violinplot of power
plt.figure(figsize=(15,7))
# Adjust the subplot's width
plt.subplots_adjust(wspace=0.2)
# Create the violinplot using Seaborn's violinplot function
sns.violinplot(x=dayCode1, y=pwr1, data=data, color='purple')
# Label the x-axis
plt.xlabel('DayCode1', fontsize=12)
# Add a title to the plot
plt.title('Violin plot of Power', fontsize=14)
# Remove the top and right spines of the plot
sns.despine(left=True, bottom=True)
# Add a tight layout to the plot
plt.tight_layout()


# Plotting the histogram and normal probability plot for 'Power' column
plt.figure(figsize=(15,7))
# Histogram of 'Power' column
plt.subplot(1,2,1)
pwr1.hist(bins=70, color='purple')
plt.title('Power Distribution', fontsize=16)

# # Normal Probability Plot of 'Power' column
# plt.subplot(1,2,2)
# # Create the normal probability plot using stats.probplot
# stats.probplot(pwr1, plot=plt, fit=True, rvalue=True)
# # Add a line to the plot
# plt.plot([0, max(pwr1)], [0, max(pwr1)], color='purple', linestyle='--')
# plt.title('Normal Probability Plot Power', fontsize=14)

# Printing the summary statistics of 'Power' column
df= data.describe().T
# print(df.to_string())

plt.show()

#Modelling and Evaluation
#Transform the Power column of the data DataFrame into a numpy array of float values

dataset = pwr1.values.astype('float32')
#Reshape the numpy array into a 2D array with 1 column

dataset = np.reshape(dataset, (-1, 1))
# print(dataset)


#Create an instance of the MinMaxScaler class to scale the values between 0 and 1

scaler = MinMaxScaler(feature_range=(0, 1),copy=True, clip=False)
# Fit the MinMaxScaler to the transformed data and transform the values
dataset = scaler.fit_transform(dataset)

#Split the transformed data into a training set (80%) and a test set (20%)
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size+1:len(dataset),:]


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# reshape into X=t and Y=t+1
look_back = 30
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

X_train.shape

Y_train.shape


# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

X_train.shape

# print(X_train.shape)

# plt.show()

# Defining the LSTM model
model = models.Sequential()
# Adding the first layer with 100 LSTM units and input shape of the data
model.add(layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
# Adding a dropout layer to avoid overfitting
model.add(layers.Dropout(0.2))
# Adding a dense layer with 1 unit to make predictions
model.add(layers.Dense(1))
# Compiling the model with mean squared error as the loss function and using Adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam')
# Fitting the model on training data and using early stopping to avoid overfitting
# history = model.fit(X_train, Y_train, epochs=100, batch_size=1240, validation_data=(X_test, Y_test),
#                     callbacks=[EarlyStopping(monitor='val_loss', patience=4)], verbose=1, shuffle=False)

history = model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_data=(X_test, Y_test), verbose=1, shuffle=False)


# Displaying a summary of the model
model.summary()
# make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# invert predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])


# Define time steps and forecast steps
print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
# plt.show();

# Define time steps and forecast steps
time_steps = 96 # 15 minutes per time step
forecast_steps = 48 # 12 hours ahead

aa=[x for x in range(48)]
# Creating a figure object with desired figure size
plt.figure(figsize=(20,6))
# Plotting the actual values in blue with a dot marker
plt.plot(aa, Y_test[0][:48], marker='.', label="actual", color='purple')
# Plotting the predicted values in green with a solid line
plt.plot(aa, test_predict[:,0][:48], '-', label="prediction", color='red')
# Removing the top spines
sns.despine(top=True)
# Adjusting the subplot location
plt.subplots_adjust(left=0.07)
# Labeling the y-axis
plt.ylabel('Power', size=14)
# Labeling the x-axis
plt.xlabel('Time step', size=14)
# Adding a legend with font size of 15
plt.legend(fontsize=16)
# Display the plot
plt.show()
