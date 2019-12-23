import numpy as np
import pandas as pd
from keras import backend as K
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_log_error
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.ops.gen_math_ops import log1p


def read_data(input, is_dataframe=False, one_hot=True):
    if not is_dataframe:
        df = pd.read_csv(input)
    else:
        df = input
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
    # df['windspeed'] = df.windspeed.astype('category')
    # df['humidity'] = df.humidity.astype('category')

    columns = list(df.columns)
    columns_to_remove = ['atemp']
    # #------------------- try to calculate casual and registered and predict their sum -------------------------
    # if 'count' in columns:
    #     columns_to_remove.extend(['count'])

    df = df.drop(columns_to_remove, axis=1)

    if one_hot:
        one_hot_columns = list(df.columns)  # getting all columns
        non_categorical_columns = ['temp', 'count', 'windspeed', 'humidity', 'casual',
                                   'registered']  # these are not categorical columns
        one_hot_columns = [x for x in one_hot_columns if x not in non_categorical_columns]  # excluding non-cat columns
        for column in one_hot_columns:
            df = pd.concat([df.drop(column, axis=1), pd.get_dummies(df[column], prefix=column)],
                           axis=1)  # creating one hot encoded columns, adding them to dataset, removing original column

    return df


# removes unwanted columns and selects one-hot encoded versions of wanted ones
def select_train_columns(df, train_columns=None, pred_column=None):
    if pred_column is None:
        pred_column = 'count'
    all_columns = list(df.columns)
    if train_columns is None:  # if we want, we can specify which columns we want
        train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather',
                         'temp', 'humidity',
                         'windspeed']
    X = df[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
    if pred_column in all_columns:  # if used for train set, we need to return the results too
        y = df[pred_column]
    else:
        y = None
    return X, y


# Normalizes the given column
def normalizer(column, df):
    x = df[[column]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)

    for i in range(len(x_scaled)):
        if x_scaled[i] <= 0.25:
            x_scaled[i] = int(0)
        elif x_scaled[i] <= 0.5:
            x_scaled[i] = int(1)
        elif x_scaled[i] <= 0.75:
            x_scaled[i] = int(2)
        else:
            x_scaled[i] = int(3)
    return x_scaled


def transform_list_item(list):
    return list[0]


def bring_to_zero(list):  # negative rental numbers don't exist, so we set them to 0
    for i, y in enumerate(list):
        if list[i] < 0:
            list[i] = 0


def sequential_nn_model(X_train, y_train):
    model = Sequential([
        Dense(100, activation='relu', input_shape=(X_train.shape[1],)),

        Dense(40, activation='relu'),

        Dense(20, activation='relu'),

        Dense(1, activation='relu')
    ])
    model.compile(optimizer='adam',
                  loss=rmsle,
                  metrics=['mean_squared_logarithmic_error'])

    hist = model.fit(X_train, y_train, epochs=50)
    return model


def create_submission(predictions):
    submission = pd.DataFrame()
    submission['Id'] = range(len(predictions))
    submission['Predicted'] = predictions
    submission.to_csv('submission.csv', index=False)


def print_scores(name, test_set, predictions):
    print('RMSLE for', name, ':', np.sqrt(mean_squared_log_error(test_set, predictions)))
    print('R2 for', name, ':', r2_score(test_set, predictions), '\n')


def rmsle(y, y0):
    return K.sqrt(K.mean(K.square(log1p(y) - log1p(y0))))
