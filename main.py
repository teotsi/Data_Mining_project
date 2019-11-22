import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib inline
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
print(df_train.head(10))

# print(df_train.shape)

df_test = pd.read_csv('test.csv')
# print(df_test.head(10))

# print (df_test.shape)

df_train = df_train.drop(['atemp', 'casual', 'registered', 'windspeed'], axis=1)
# print(df_train.head(10))

df_train.rename(columns={'weathersit': 'weather',  # renaming columns to improve readability
                         'mnth': 'month',
                         'hr': 'hour',
                         'yr': 'year',
                         'hum': 'humidity',
                         'cnt': 'count'}, inplace=True)

one_hot_columns = list(df_train.columns)  # getting all columns
pure_columns = ['temp', 'humidity', 'count']  # these are not categorical columns
one_hot_columns = [x for x in one_hot_columns if x not in pure_columns]  # excluding non-cat columns
for column in one_hot_columns:
    print("yikes")
    df_train = pd.concat([df_train.drop(column, axis=1), pd.get_dummies(df_train[column], prefix=column)],
                         axis=1)  # creating one hot encoded columns, adding them to dataset, removing original column

print(df_train.head(10))
# Training
all_columns = list(df_train.columns)
# Training and test data is created by splitting the main data. 30% of test data is considered
train_columns = ['temp', 'humidity', 'workingday']
X = df_train[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
y = df_train['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ->Prediction
clf = LinearRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print(y_pred)
