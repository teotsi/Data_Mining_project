import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics.regression import mean_squared_log_error, r2_score
from sklearn.model_selection import GridSearchCV
# matplotlib inline
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor



pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)

def read_data(filename):
    df = pd.read_csv(filename)
    df.rename(columns={'weathersit': 'weather',
                       'mnth': 'month',
                       'hr': 'hour',
                       'yr': 'year',
                       'hum': 'humidity',
                       'cnt': 'count'}, inplace=True)
    df['season'] = df.season.astype('category')
    df['year'] = df.year.astype('category')
    df['month'] = df.month.astype('category')
    df['hour'] = df.hour.astype('category')
    df['holiday'] = df.holiday.astype('category')
    df['weekday'] = df.weekday.astype('category')
    df['workingday'] = df.workingday.astype('category')
    df['weather'] = df.weather.astype('category')
    columns = list(df.columns)
    columns_to_remove = ['atemp', 'windspeed']
    if 'casual' in columns:
        columns_to_remove.extend(['casual', 'registered'])
    df = df.drop(columns_to_remove, axis=1)
    one_hot_columns = list(df.columns)  # getting all columns
    non_categorical_columns = ['temp', 'humidity', 'count']  # these are not categorical columns
    one_hot_columns = [x for x in one_hot_columns if x not in non_categorical_columns]  # excluding non-cat columns
    for column in one_hot_columns:
        df = pd.concat([df.drop(column, axis=1), pd.get_dummies(df[column], prefix=column)],
                       axis=1)  # creating one hot encoded columns, adding them to dataset, removing original column

    return df


filename = 'train.csv'
df_train = read_data(filename)
# print(df_train.head(10))

df_test = read_data('test.csv')  # reading test file
# print(df_test.head(10))

# print(df_train.head(10))
#print(df_train.head(10))
# Training
all_columns = list(df_train.columns)
# Training and test data is created by splitting the main data. 30% of test data is considered
train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity']
X = df_train[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
# print(X)
y = df_train['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

reg = Lasso(alpha=0.03, copy_X=True, fit_intercept=False, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
reg.fit(X,y)
prediction = reg.predict(X_test)
print(reg.score(X_test, y_test))

reg_bag = BaggingRegressor(reg)
reg_bag.fit(X,y)


params = {'n_estimators': [10,15,20]}
#
# 'selection': ['cyclic', 'random'],
#           'tol': [i/100 for i in range(1,10)]
reg_test = GridSearchCV(estimator=reg_bag, cv=5 , param_grid = params, n_jobs=-1)
reg_test.fit(X,y)

# print('Best precompute:', reg_test.best_estimator_.precompute)
# print('Best alpha:', reg_test.best_estimator_.alpha)
# print('Best selection:', reg_test.best_estimator_.selection)
# print('Best tol:', reg_test.best_estimator_.tol)

y_pred = reg_test.predict(X_test)
# print(y_pred)
for i, y in enumerate(y_pred):
    if y_pred[i] < 0:
        y_pred[i] = 0

print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))