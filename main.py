#%%

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

filename = 'train.csv'
df_train =pd.DataFrame(pd.read_csv(filename))
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)

df_train.rename(columns={'weathersit':'weather',
                     'mnth':'month',
                     'hr':'hour',
                     'yr':'year',
                     'hum': 'humidity',
                     'cnt':'count'},inplace=True)

# df_train['season'] = df_train.season.astype('category')
# df_train['year'] = df_train.year.astype('category')
# df_train['month'] = df_train.month.astype('category')
# df_train['hour'] = df_train.hour.astype('category')
# df_train['holiday'] = df_train.holiday.astype('category')
# df_train['weekday'] = df_train.weekday.astype('category')
# df_train['workingday'] = df_train.workingday.astype('category')
# df_train['weather'] = df_train.weather.astype('category')


# #print(df_train.dtypes)
# fig,[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8] = plt.subplots(nrows=8, figsize=(15,25))
# sn.barplot(x = df_train['season'], y = df_train['count'],ax = ax1)
# sn.barplot(x = df_train['year'], y = df_train['count'],ax = ax2)
# sn.barplot(x = df_train['month'], y = df_train['count'],ax = ax3)
# sn.barplot(x = df_train['hour'], y = df_train['count'],ax = ax4)
# sn.barplot(x = df_train['holiday'], y = df_train['count'],ax = ax5)
# sn.barplot(x = df_train['weekday'], y = df_train['count'],ax = ax6)
# sn.barplot(x = df_train['workingday'], y = df_train['count'],ax = ax7)
# sn.barplot(x = df_train['weather'], y = df_train['count'],ax = ax8)


# # Regression plot is used to verify if a pattern can be observed between `count` and numerical variables
# fig,[ax1,ax2,ax3] = plt.subplots(ncols = 3, figsize = (20,8))
# plt.rc('xtick', labelsize=10) 
# plt.rc('ytick', labelsize=10) 

# sn.regplot(x = 'temp', y = 'count',data = df_train, ax = ax1)
# ax1.set(title="Relation between temperature and count")
# sn.regplot(x = 'humidity', y = 'count',data = df_train, ax = ax2)
# ax2.set(title="Relation between humidity and total count")
# sn.regplot(x = 'windspeed', y = 'count',data = df_train, ax = ax3)
# ax3.set(title="Relation between windspeed and count")


# #Correlation analysis
# data_corr = df_train[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']].corr()
# mask = np.array(data_corr)
# mask[np.tril_indices_from(mask)] = False
# fig = plt.subplots(figsize=(15,10))
# sn.heatmap(data_corr, mask=mask, vmax=1, square=True, annot=True, cmap="YlGnBu")

# Training and test data is created by splitting the main data. 30% of test data is considered
X = df_train[['temp', 'humidity', 'workingday']]
y = df_train['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LinearRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)

# %%
