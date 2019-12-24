import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import Reader
import pandas as pd 

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)
if os.path.exists('log.txt'):
    append_write = 'a'  # append if already exists
else:
    append_write = 'w'  # make a new file if not

predict_columns = ['casual', 'registered', 'count']

merged_pred = {}  # dictionary with predictions for each column
merged_pred_df = {}  # dictionary with dataframes
mean_pred = {}  # dictionary with mean results
for column in predict_columns:
    print("Training for " + column + '\n')
    with open('log.txt', append_write) as file:
        if append_write == 'w':
            append_write = 'a'
        file.write('Training for ' + column)
    df = Reader.read_data('train.csv', one_hot=False, extra_csv=column+'.csv')
    df_test = Reader.read_data('test.csv', one_hot=False, extra_csv=column+'.csv')  # random forest is used without one-hot encoding

    X, y = Reader.select_train_columns(df, pred_column=column)

    # Training and test data is created by splitting the main data. 30% of test data is considered
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    df_test = Reader.select_train_columns(df_test, pred_column=column)[0]  # getting the correctly encoded parameters
    merged_pred[column] = []

    # ----- using Random Forest -----
    for i in range(20):
        rf = RandomForestRegressor(n_jobs=-1, max_depth=75, n_estimators=900, random_state=0)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_pred = Reader.bring_to_zero(rf_pred)
        merged_pred[column].append(pd.Series(rf_pred, name='pred_rf' + str(i)))
    scores = Reader.get_scores(column + " RF", merged_pred[column][0], y_test)
    with open('log.txt', append_write) as file:
        file.write(scores)
    print(scores)
    # reading data again to apply one-hot encoding

    df = Reader.read_data('train.csv', extra_csv=column+'.csv')
    df_test = Reader.read_data('test.csv', extra_csv=column+'.csv')

    X, y = Reader.select_train_columns(df, pred_column=column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # splitting

    df_test['weather_4'] = 0
    df_test = Reader.select_train_columns(df_test, pred_column=column)[0]

    # ------- using neural network -----
    for i in range(30):
        nn = Reader.sequential_nn_model(X_train, y_train)  # fitting neural network model on X and y
        nn_pred = nn.predict(X_test)  # making prediction
        nn_pred = [Reader.transform_list_item(x) for x in nn_pred]
        merged_pred[column].append(pd.Series(nn_pred, name='pred_nn' + str(i)))
    scores = Reader.get_scores(column + " Neural Network", y_test, nn_pred)
    with open('log.txt', append_write) as file:
        file.write(scores)
    print(scores)
    # ------- using mlp -------
    for i in range(10):
        mlp = MLPRegressor(hidden_layer_sizes=(100, 60, 40, 20), activation='relu', solver='lbfgs', alpha=0.0001,
                           verbose=False,
                           max_iter=400)
        mlp.fit(X_train, y_train)
        mlp_pred = mlp.predict(X_test)
        mlp_pred = Reader.bring_to_zero(mlp_pred)
        merged_pred[column].append(pd.Series(mlp_pred, name='pred_mlp' + str(i)))

    # merging the results from each method in a single dataframe
    merged_pred_df[column] = pd.concat(merged_pred[column], axis=1)
    print(merged_pred_df[column].head())
    # getting the mean
    mean_pred[column] = pd.DataFrame()
    mean_pred[column]['avg'] = merged_pred_df[column].mean(axis=1)  # getting the mean average of the columns
    scores = Reader.get_scores(column + " AVG prediction", y_test, mean_pred[column]['avg'].tolist())
    with open('log.txt', append_write) as file:
        file.write(scores)
    print(scores)
# now we are going to merge casual and registered predictions
summed_pred = mean_pred['casual'] + mean_pred['registered']
scores = Reader.get_scores("Summed prediction", y_test, summed_pred)
with open('log.txt', append_write) as file:
    file.write(scores)
print(scores)
# create_submission(summed_pred, filename='summed_submission.csv')
