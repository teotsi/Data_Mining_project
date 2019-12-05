import pandas as pd


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


def transform_list(list):
    return list[0]
