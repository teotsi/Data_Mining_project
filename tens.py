from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.models import Sequential
import pandas as pd
from Reader import *


def sequential_nn_model(X_train, y_train):
    model = Sequential([
        Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(40, activation='relu'),
        Dense(20, activation='relu'),
        Dense(1, activation='relu')
    ])
    model.compile(optimizer='adam',
                  loss='mean_squared_logarithmic_error',
                  metrics=['mean_squared_logarithmic_error'])

    hist = model.fit(X_train, y_train, epochs=100)
    return model

if __name__=='__main__':

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 13000)

    filename = 'train.csv'
    df_train = read_data(filename)

    df_test = read_data('test.csv')  # reading test file

    # Training
    all_columns = list(df_train.columns)
    # Training and test data is created by splitting the main data. 30% of test data is considered
    train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity']
    X = df_train[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
    # print(X)
    X = X.drop(columns=['weather_4'])
    print(X.columns)
    y = df_train['count']

    # Creating the split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initializing MLPRegressor
    neural = MLPRegressor(hidden_layer_sizes=(100, 60, 40, 20), activation='relu', solver='lbfgs', alpha=0.0001,
                          verbose=True)
    neural.fit(X_train, y_train)

    _, test_acc = neural.evaluate(X_test, y_test, verbose=0)  # evaluating MLPRegressor
    print('Test: %.3f' % test_acc)

    # Initializing Sequential NN
    model = sequential_nn_model(X_train, y_train)

    df_test['weather_4'] = 0
    df_test = df_test[[x for x in all_columns if x.startswith(tuple(train_columns))]]
    df_test = df_test.drop(columns=['weather_4'])
    print(df_test.columns)
    test_array = df_test.to_numpy()
    predictions = model.predict(test_array)
    individual_predictions = [transform_list(x) for x in predictions]
    for i, y in enumerate(individual_predictions):
        if individual_predictions[i] < 0:
            individual_predictions[i] = 0

    # mlp_predictions=neural.predict(df_test_og)
    # for i, y in enumerate(mlp_predictions):
    #     if mlp_predictions[i] < 0:
    #         mlp_predictions[i] = 0

    # full_set = dstack((individual_predictions,mlp_predictions))
    # full_set = full_set.reshape(full_set.shape[0],full_set.shape[1]*full_set.shape[2])
    # print(full_set.shape)
    # ensemble_model = LogisticRegression()
    # ensemble_model.fit(full_set, y)
    #
    # final_prediction = ensemble_model.predict(test_array)

    submission = pd.DataFrame()
    submission['Id'] = range(len(individual_predictions))
    submission['Predicted'] = individual_predictions
    submission.to_csv("submission.csv", index=False)
