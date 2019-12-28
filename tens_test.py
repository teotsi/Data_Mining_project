import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.layers.recurrent import LSTM
from tensorflow_core.python.keras.models import Sequential
from sklearn.metrics import mean_squared_log_error, r2_score
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)


def transform_list(list):
    return list[0]


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

df_test = read_data('test.csv')  # reading test file

# Training
all_columns = list(df_train.columns)
# Training and test data is created by splitting the main data. 30% of test data is considered
train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity']
X = df_train[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
# print(X)
y = df_train['count']

# Creating the split
X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5, random_state=42)
print(X_test.head(10))
# print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

# model = Sequential([
#     Dense(100, activation='relu', input_shape=(57,)),
#     Dense(40, activation='relu'),
#     Dense(20, activation='relu'),
#     Dense(1, activation='relu')
# ])
# model.compile(optimizer='adam',
#               loss='mean_squared_logarithmic_error',
#               metrics=['mean_squared_logarithmic_error'])

# hist = model.fit(X, y, epochs=100)
NNmodel = Sequential()

NNmodel.add(Dense(57, kernel_initializer='normal', activation='relu'))

NNmodel.add(Dense(100, kernel_initializer='normal', activation='relu'))
NNmodel.add(Dense(40, kernel_initializer='normal', activation='relu'))
NNmodel.add(Dense(20, kernel_initializer='normal', activation='relu'))

NNmodel.add(Dense(1, kernel_initializer='normal', activation='relu'))

NNmodel.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['mean_squared_logarithmic_error'])

df_test['weather_4'] = 0
df_test = df_test[[x for x in all_columns if x.startswith(tuple(train_columns))]]
test_array = df_test.to_numpy()
predictions = NNmodel.predict(test_array)


print('hello')

individual_predictions = [transform_list(x) for x in predictions]
for i, y in enumerate(individual_predictions):
    if individual_predictions[i] < 0:
        individual_predictions[i] = 0

submission = pd.DataFrame()
submission['Id'] = range(len(individual_predictions))
submission['Predicted'] = individual_predictions
submission.to_csv("submission.csv", index=False)
