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
# # Calculate summary statistics by occupation
# stats = df.groupby('DayCode1').agg({'Occupation1': ['mean', 'std']})
# stats.columns = [' '.join(col).strip() for col in stats.columns.values]
#
# # Create a normal plot using seaborn
# sns.displot(df, x='Power1', hue='Occupation1', kind='kde', fill=True)
#
# # Show the plot
# plt.show()

# Calculate summary statistics by occupation
stats = df.groupby('DayCode').agg({'Occupation': ['mean', 'std']})
stats.columns = [' '.join(col).strip() for col in stats.columns.values]
# Create a normal plot using seaborn
sns.displot(df, x='Power', hue='Occupation', kind='kde', fill=True)
# Show the plot
plt.show()



