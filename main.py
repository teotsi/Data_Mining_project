

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib inline
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, HuberRegressor, ElasticNet, ElasticNetCV, \
    Lars, LarsCV, LogisticRegressionCV
from sklearn.metrics.regression import mean_squared_log_error, r2_score

filename = 'Data_Mining_project/train.csv'
df_train = pd.read_csv(filename)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)

df_train.rename(columns={'weathersit':'weather',
                     'mnth':'month',
                     'hr':'hour',
                     'yr':'year',
                     'hum': 'humidity',
                     'cnt':'count'},inplace=True)

#THESE ARE THE PARAMETERS THAT WILL BE ONE-HOT ENCODED

df_train['season'] = df_train.season.astype('category')
df_train['year'] = df_train.year.astype('category')
df_train['month'] = df_train.month.astype('category')
df_train['hour'] = df_train.hour.astype('category')
df_train['holiday'] = df_train.holiday.astype('category')
df_train['weekday'] = df_train.weekday.astype('category')
df_train['workingday'] = df_train.workingday.astype('category')
df_train['weather'] = df_train.weather.astype('category')

print(df_train.head(10))



df_test = pd.read_csv('Data_Mining_project/test.csv') #reading test file
df_test.rename(columns={'weathersit':'weather', #renaming test columns
                     'mnth':'month',
                     'hr':'hour',
                     'yr':'year',
                     'hum': 'humidity',
                     'cnt':'count'},inplace=True)

df_test['season'] = df_test.season.astype('category')
df_test['year'] = df_test.year.astype('category')
df_test['month'] = df_test.month.astype('category')
df_test['hour'] = df_test.hour.astype('category')
df_test['holiday'] = df_test.holiday.astype('category')
df_test['weekday'] = df_test.weekday.astype('category')
df_test['workingday'] = df_test.workingday.astype('category')
df_test['weather'] = df_test.weather.astype('category')

df_test = df_test.drop(['atemp', 'windspeed'], axis=1) #dropping cols
print(df_test.weather.unique())

one_hot_columns = list(df_test.columns)  # getting all columns
non_categorical_columns = ['temp', 'humidity', 'count']  # these are not categorical columns
one_hot_columns = [x for x in one_hot_columns if x not in non_categorical_columns]  # excluding non-cat columns
for column in one_hot_columns:
    df_test = pd.concat([df_test.drop(column, axis=1), pd.get_dummies(df_test[column], prefix=column)],
                         axis=1)  # creating one hot encoded columns, adding them to dataset, removing original column

df_train = df_train.drop(['atemp', 'casual', 'registered', 'windspeed'], axis=1)
# print(df_train.head(10))


one_hot_columns = list(df_train.columns)  # getting all columns
non_categorical_columns = ['temp', 'humidity', 'count']  # these are not categorical columns
one_hot_columns = [x for x in one_hot_columns if x not in non_categorical_columns]  # excluding non-cat columns
for column in one_hot_columns:
    df_train = pd.concat([df_train.drop(column, axis=1), pd.get_dummies(df_train[column], prefix=column)],
                         axis=1)  # creating one hot encoded columns, adding them to dataset, removing original column

print(df_train.head(10))
# Training
all_columns = list(df_train.columns)
# Training and test data is created by splitting the main data. 30% of test data is considered
train_columns = ['season','month','hour','holiday','weekday','workingday','weather','temp', 'humidity']
X = df_train[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
# print(X)
y = df_train['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ->Prediction
clf = LogisticRegression(n_jobs=-1, solver="newton-cg")
clf.fit(X, y)
y_pred = clf.predict(X_test)
# print(y_pred)
for i, y  in enumerate(y_pred):
    if y_pred[i] < 0:
        y_pred[i] = 0

print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))

# print (df_test.head(5))
df_test['weather_4']=0
df_test = df_test[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
y_pred = clf.predict(df_test)
for i, y  in enumerate(y_pred):
    if y_pred[i] < 0:
        y_pred[i] = 0

submission = pd.DataFrame()
submission['Id'] = range(y_pred.shape[0])
submission['Predicted'] = y_pred
submission.to_csv("submission.csv", index=False)