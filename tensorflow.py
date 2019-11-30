import numpy as np
import pandas as pd 
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense

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


df_test = read_data('test.csv')  # reading test file

# Training
all_columns = list(df_train.columns)
# Training and test data is created by splitting the main data. 30% of test data is considered
train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity']
X = df_train[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
# print(X)
y = df_train['count']


#Creating the split
X_train = X.sample(frac=0.8, random_state=42)
X_test = X.drop(X_train.index)

y_train = y.sample(frac=0.8, random_state=42)
y_test = y.drop(y_train.index)

model = keras.Sequential([
    keras.layers.Dense(100, activation="relu", input_shape=(X_train.shape())),
    keras.layers.Dense(40, activation='relu'),
    keras.layers.Dense(20, activation='relu')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train,y_train, epochs=5)

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Tested:', test_acc)