import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sn
from scipy import stats
from numpy import median
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

filename = 'train.csv'
df_train = pd.read_csv(filename)
#print(df_train.head(10))

#print(df_train.shape)

df_test = pd.read_csv('test.csv')
#print(df_test.head(10))

#print (df_test.shape)

df_train = df_train.drop(['atemp', 'casual', 'registered', 'windspeed'], axis=1)
#print(df_train.head(10))

df_train.rename(columns={'weathersit':'weather',
                     'mnth':'month',
                     'hr':'hour',
                     'yr':'year',
                     'hum': 'humidity',
                     'cnt':'count'},inplace=True)
#Training

# Training and test data is created by splitting the main data. 30% of test data is considered
X = df_train[['temp', 'humidity', 'workingday']]
y = df_train['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#->Prediction
clf = LinearRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)