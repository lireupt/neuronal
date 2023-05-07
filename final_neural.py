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
df = pd.read_csv('LCAlgarve2.csv', parse_dates=True, index_col='DayCode1')


#Global dataset by array  variables
# dayCode1    = df["DayCode1"]
pwr1        = df['Power1']
occ1        = df["Occupation1"]

#Sort day code by ascending
dayCodeSort =df.sort_values(by="DayCode1", ascending= True)
print(dayCodeSort)

#Align dataset
#
# # Print the number of rows and columns in the data
# print('Number of rows and columns:', df.shape)
# print(df.head(5))

#
# # Get the information about the dataframe
# print("\nInformation about the dfframe:")
# print(df.info())
#
# # Get the data type of each column in the dataframe
# print("\nData type of each column in the dataframe:")
# print(df.dtypes)

#Convert data type to float in columns Occupation2
df['Occupation1'] = pd.to_numeric(df['Occupation1'], errors='coerce', downcast="integer")
# df['Occupation2'] = pd.to_numeric(df['Occupation2'], errors='coerce', downcast='float')
print(df.dtypes)
print(df)

#Testing for Normality
#We will use D’Agostino’s K^2 Test to determine if our data is normally distributed.
#In the SciPy implementation of the test, the p-value will be used to make the following interpretation:
stat, p = stats.normaltest(df.Power1)
# print('Statistics=%.3f, p=%.3f' % (stat, p))

# Set the significance level
alpha = 0.05

# Make a decision on the test result
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')

print( 'Kurtosis of normal distribution: {}'.format(stats.kurtosis(df.Power1)))
print( 'Skewness of normal distribution: {}'.format(stats.skew(df.Power1)))
sns.histplot(df["Power1"], kde=True, stat="density")
plt.show()
sns.histplot(dayCodeSort,color='purple')


#Plot from Consumption for Dataset
# Exploratory Data Analysis(EDA)
plt.figure(figsize=(14,6))
# plt.plot(df['Power1'], color='purple')
plt.ylabel('Global Active Power', fontsize=12)
plt.xlabel('Date', fontsize=12)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
plt.tick_params(bottom = False)
plt.title('Active Power Consumption for Dataset', fontsize=14)
plt.tight_layout()
plt.grid(True)
sns.despine(bottom=True, left=True)
# plt.show()



# Plot the first subplot showing the violinplot of global active power
# Adjust the subplot's width
plt.subplots_adjust(wspace=0.2)
# Create the violinplot using Seaborn's violinplot function
# sns.violinplot(x=dayCode1 , y=pwr1, df=df, color='purple')
# Label the x-axis
plt.xlabel('DayCode', fontsize=12)
# Add a title to the plot
plt.title('Violin plot of Global Active Power', fontsize=14)
# Remove the top and right spines of the plot
sns.despine(left=True, bottom=True)
# Add a tight layout to the plot
plt.tight_layout()



# Plotting the histogram and normal probability plot for 'Global_active_power' column
plt.figure(figsize=(15,7))

# Histogram of 'Global_active_power' column
plt.subplot(1,2,1)
pwr1.hist(bins=70, color='purple')
plt.title('Global Active Power Distribution', fontsize=16)

# Normal Probability Plot of 'Global_active_power' column
plt.subplot(1,2,2)
# Create the normal probability plot using stats.probplot
stats.probplot(pwr1, plot=plt, fit=True, rvalue=True)
# Add a line to the plot
plt.plot([0, max(pwr1)], [0, max(pwr1)], color='purple', linestyle='--')
plt.title('Normal Probability Plot of Global Active Power', fontsize=14)


# Printing the summary statistics of 'Global_active_power' column
df= df.describe().T
# print(df.to_string())