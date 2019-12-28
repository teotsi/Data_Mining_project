import numpy as np
import pandas as pd
from keras import backend as K
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_log_error
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.ops.gen_math_ops import log1p
from sklearn.experimental import enable_hist_gradient_boosting
import sklearn
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingRegressor


def read_data(input, is_dataframe=False, one_hot=True, extra_csv=None):
    if not is_dataframe:
        df = pd.read_csv(input)
    else:
        df = input

    df['windspeed'] = normalizer('windspeed', df, for_one_hot=False)

    if extra_csv is None:
        extra_csv = 'count.csv'
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

    df = avg_cnt_per_holiday_by_mnth(df)
    df = avg_cnt_per_day_of_month(df, extra_csv)
    df = avg_cnt_By_Year_by_mnth(df)
    df = avg_cnt_per_weekday_of_year(df)
    df = avg_casual_by_weekDay_by_Weather(df)
    df = avg_casual_perWorkingDay(df)

    columns_to_remove = ['atemp']
    # #------------------- try to calculate casual and registered and predict their sum -------------------------
    # if 'count' in columns:
    #     columns_to_remove.extend(['count'])

    df = df.drop(columns_to_remove, axis=1)

    if one_hot:
        one_hot_columns = list(df.columns)  # getting all columns
        non_categorical_columns = ['temp', 'count', 'windspeed', 'humidity', 'casual',
                                   'registered', 'Count_By_Month_of_Year_avg', 'year_day_cnt_avg',
                                   'Month_day_cnt_avg', 'Avg_casual_on_Workday', 'cnt_per_holiday_by_mnth',
                                   'Avg_casual_by_Weekday_by_Weather']  # these are not categorical columns
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
                         'Count_By_Month_of_Year_avg', 'year_day_cnt_avg',
                         'Month_day_cnt_avg', 'cnt_per_holiday_by_mnth',
                         'Avg_casual_by_Weekday_by_Weather', 'windspeed', 'Avg_casual_on_Workday']
    X = df[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
    if pred_column in all_columns:  # if used for train set, we need to return the results too
        y = df[pred_column]
    else:
        y = None
    return X, y


# Normalizes the given column
def normalizer(column, df, for_one_hot=False):
    x = df[[column]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)

    if for_one_hot:
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
    return abs(list[0])


def bring_to_zero(list):  # negative rental numbers don't exist, so we set them to 0
    for i, y in enumerate(list):
        if list[i] < 0:
            list[i] = 0
    return list


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

    hist = model.fit(X_train, y_train, epochs=50, verbose=0)
    return model


def isolation_forest(X, y, drop_outliers=True):
    clf = IsolationForest(contamination=0.05)
    clf.fit(X)
    pred_outlier = clf.predict(X)
    print('Number of outliers: ', (len(pred_outlier) - pred_outlier.sum()) / 2)
    if drop_outliers:
        for i in range(X.shape[0]):
            if pred_outlier[i] == -1:
                X = X.drop([i])
                y = y.drop([i])
    return X, y


def gradient_boost_with_extra_trees(X, y):
    gb = HistGradientBoostingRegressor(loss='least_absolute_deviation', learning_rate=0.12, max_depth=150)

    ex = ExtraTreesRegressor(n_jobs=-1, max_depth=100, n_estimators=100, random_state=0)

    rf = RandomForestRegressor(random_state=0, max_depth=25, n_estimators=900)
    done = False
    while not done:
        # try:
        stacking = sklearn.ensemble.VotingRegressor(
            estimators=[('gradientBoost', gb), ('randomForest', rf), ("extraTrees", ex)],
            n_jobs=-1)
        stacking.fit(X, y)
        done = True
    # except ValueError:
    # print("Caught ValueError")
    return stacking


def avg_cnt_per_holiday_by_mnth(df):
    extra = pd.read_csv('cnt_per_holiday_by_mnth.csv')
    cnt_per_holiday_by_mnth = []
    for i in range(df.shape[0]):
        holiday = df.holiday[i]
        mnth = df.month[i]
        cnt_per_holiday_by_mnth.append(extra.iloc[holiday][mnth - 1])
    df['cnt_per_holiday_by_mnth'] = cnt_per_holiday_by_mnth
    return df


def avg_cnt_per_weekday_of_year(df):
    extra = pd.read_csv('cntByYrByWd.csv')
    cnt_avg_peryr_per_weekday = []
    for i in range(df.shape[0]):
        weekday = df.weekday[i]
        year = df.year[i]
        cnt_avg_peryr_per_weekday.append(extra.iloc[weekday][year])
    df['year_day_cnt_avg'] = cnt_avg_peryr_per_weekday
    return df


def avg_cnt_per_day_of_month(df, extra_csv=None):
    if extra_csv is None:
        extra_csv = 'count.csv'
    extra = pd.read_csv(extra_csv)
    cnt_avg_perMnth_perDay = []
    for i in range(df.shape[0]):
        month = df.month[i]
        day = df.weekday[i]
        cnt_avg_perMnth_perDay.append(extra.iloc[month - 1][day])
    df['Month_day_cnt_avg'] = cnt_avg_perMnth_perDay
    return df


def avg_cnt_By_Year_by_mnth(df, extra_csv=None):
    if extra_csv is None:
        extra_csv = 'cntByYrByMnth.csv'
    extra = pd.read_csv(extra_csv)
    avg_cnt_By_Year_by_Month = []
    for i in range(df.shape[0]):
        year = df.year[i]
        month = df.month[i]
        if year == 0:
            avg_cnt_By_Year_by_Month.append(extra.iloc[month - 1][2])
        else:
            avg_cnt_By_Year_by_Month.append(extra.iloc[month + 10][2])
    df['Count_By_Month_of_Year_avg'] = avg_cnt_By_Year_by_Month
    return df


def avg_casual_by_weekDay_by_Weather(df, extra_csv=None):
    if extra_csv is None:
        extra_csv = 'Avg_casual_perWeekday_basedOnWeather.csv'
    extra = pd.read_csv(extra_csv)
    avg_casual_by_weekday_by_weather = []
    for i in range(df.shape[0]):
        weekday = df.weekday[i]
        weather = df.weather[i]
        avg_casual_by_weekday_by_weather.append(extra.iloc[weekday][weather - 1])
    df['Avg_casual_by_Weekday_by_Weather'] = avg_casual_by_weekday_by_weather
    return df


def avg_casual_perWorkingDay(df, extra_csv=None):
    if extra_csv is None:
        extra_csv = 'Avg_casual_perWorkingDay.csv'
    extra = pd.read_csv(extra_csv)
    avg_casual_perWorkDay = []
    for i in range(df.shape[0]):
        workingDay = df.workingday[i]
        avg_casual_perWorkDay.append(extra.iloc[workingDay][1])
    df['Avg_casual_on_Workday'] = avg_casual_perWorkDay
    return df


def create_submission(predictions, filename=None):
    if filename is None:
        filename = 'submission.csv'
    submission = pd.DataFrame()
    submission['Id'] = range(len(predictions))
    submission['Predicted'] = predictions
    submission.to_csv(filename, index=False)


def get_scores(name, test_set, predictions):
    string = '\n'
    string += 'RMSLE for ' + name + ': ' + str(np.sqrt(mean_squared_log_error(test_set, predictions)))
    string += '\n' + 'R2 for ' + name + ': ' + str(r2_score(test_set, predictions)) + '\n'
    return string


def log_scores(scores):
    with open('log.txt', 'a') as file:
        file.write(scores)
    print(scores)


def rmsle(y, y0):
    return K.sqrt(K.mean(K.square(log1p(y) - log1p(y0))))
