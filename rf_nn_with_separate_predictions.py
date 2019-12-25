import os
from copy import deepcopy

from numpy import mean
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from Reader import *

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)
if os.path.exists('log.txt'):
    append_write = 'a'  # append if already exists
else:
    append_write = 'w'  # make a new file if not

predict_columns = ['casual', 'registered','count']

merged_pred = {}  # dictionary with predictions for each column
merged_pred_df = {}  # dictionary with dataframes
mean_pred = {}  # dictionary with mean results
y_tests = {}
for column in predict_columns:
    print("Training for " + column + '\n')
    with open('log.txt', append_write) as file:
        if append_write == 'w':
            append_write = 'a'
        file.write('Training for ' + column)
    df = read_data('train.csv', one_hot=False, extra_csv=column + '.csv')
    df_test = read_data('test.csv', one_hot=False,
                        extra_csv=column + '.csv')  # random forest is used without one-hot encoding

    X, y = select_train_columns(df, pred_column=column)


    # Training and test data is created by splitting the main data. 30% of test data is considered
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_tests[column] = deepcopy(y_test)
    df_test = select_train_columns(df_test, pred_column=column)[0]  # getting the correctly encoded parameters
    merged_pred[column] = []

    if column == 'casual':
        casual_model = gradient_boost_with_random_forest(X_train, y_train)
        casual_y_pred = casual_model.predict(X_test)
        print(get_scores("gradient rf boost for casual", y_test,bring_to_zero(casual_y_pred)))

    # ----- using Random Forest -----
    for i in range(1):
        rf = RandomForestRegressor(n_jobs=-1, max_depth=75, n_estimators=900, random_state=0)
        rf.fit(X_train,y_train)
        rf_pred = rf.predict(X_test)
        rf_pred = bring_to_zero(rf_pred)
        merged_pred[column].append(pd.Series(rf_pred, name='pred_rf' + str(i)))

    scores = get_scores(column + " RF", merged_pred[column][0], y_test)
    with open('log.txt', append_write) as file:
        file.write(scores)
    print(scores)

    # Using gradient boost with random forest
    for i in range(1):
        gbrf = gradient_boost_with_random_forest(X_train, y_train)
        pred_gbrf = gbrf.predict(X_test)
        pred_gbrf = bring_to_zero(pred_gbrf)
        merged_pred[column].append(pd.Series(pred_gbrf, name='pred_gbrf' + str(i)))
    scores = get_scores(column + " gbrf prediction", y_test, pred_gbrf)
    with open('log.txt', append_write) as file:
        file.write(scores)
    print(scores)

    # reading data again to apply one-hot encoding

    df = read_data('train.csv', extra_csv=column + '.csv')
    df_test = read_data('test.csv', extra_csv=column + '.csv')

    X, y = select_train_columns(df, pred_column=column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # splitting

    df_test['weather_4'] = 0
    df_test = select_train_columns(df_test, pred_column=column)[0]

    # ------- using neural network -----
    for i in range(1):
        nn = sequential_nn_model(X_train,y_train)  # fitting neural network model on X and y
        nn_pred = nn.predict(X_test)  # making prediction
        nn_pred = [transform_list_item(x) for x in nn_pred]
        merged_pred[column].append(pd.Series(nn_pred, name='pred_nn' + str(i)))
    scores = get_scores(column + " Neural Network", y_test, nn_pred)
    with open('log.txt', append_write) as file:
        file.write(scores)
    print(scores)
    # ------- using mlp -------
    for i in range(0):
        mlp = MLPRegressor(hidden_layer_sizes=(100, 60, 40, 20), activation='relu', solver='lbfgs', alpha=0.0001,
                           verbose=False,
                           max_iter=400)
        mlp.fit(X,y)
        mlp_pred = mlp.predict(df_test)
        mlp_pred = bring_to_zero(mlp_pred)
        merged_pred[column].append(pd.Series(mlp_pred, name='pred_mlp' + str(i)))

    # merging the results from each method in a single dataframe
    merged_pred_df[column] = pd.concat(merged_pred[column], axis=1)
    print(merged_pred_df[column].head())
    # getting the mean
    mean_pred[column] = pd.DataFrame()
    mean_pred[column]['avg'] = merged_pred_df[column].mean(axis=1)  # getting the mean average of the columns
    scores = get_scores(column + " AVG prediction", y_test, bring_to_zero(mean_pred[column]['avg'].tolist()))
    with open('log.txt', append_write) as file:
        file.write(scores)
    print(scores)
    mean = np.mean(y_test - mean_pred[column]['avg'].tolist())
    print(mean)
    mean_list = mean_pred[column]['avg'].tolist()
    mean_list = [x + (mean / 2) for x in mean_list]
    mean_list=bring_to_zero(mean_list)
    scores = get_scores('adjusted predictions for '+ column, y_test, mean_list)
    with open('log.txt', append_write) as file:
        file.write(scores)
    print(scores)

# now we are going to merge casual and registered predictions
summed_pred = mean_pred['casual'] + mean_pred['registered']

scores = get_scores("Summed prediction", y_test, summed_pred)
with open('log.txt', append_write) as file:
    file.write(scores)
print(scores)
summed_pred_array = summed_pred.to_numpy()
summed_pred_array = np.floor(summed_pred_array)
scores = get_scores('Floor summed predictions', y_test, summed_pred_array)
with open('log.txt', append_write) as file:
    file.write(scores)
print(scores)

summed_pred = mean_pred['registered'].to_numpy() + casual_y_pred
summed_pred = np.floor(summed_pred)
print(summed_pred)
scores = get_scores('Floor summed gradient casual predictions', y_test, summed_pred)
with open('log.txt', append_write) as file:
    file.write(scores)
print(scores)

# create_submission(summed_pred_array, filename='summed_submission_15nn.csv')

#
# print("now we are going to adjust everything")
# mean = np.mean(y_test - summed_pred['avg'].values.tolist())
# print(mean)
# mean_list = summed_pred['avg'].values.tolist()
# mean_list = [x + (mean/2) for x in bring_to_zero(mean_list)]
# scores = get_scores('adjusted predictions for summed', y_test, mean_list)
# with open('log.txt', append_write) as file:
#     file.write(scores)
# print(scores)
#
# # adjusting casual
# y_test_casual = y_tests['casual'].values.tolist()
# casual_mean_list = bring_to_zero(mean_pred['casual']['avg'].values.tolist())
# print(len(y_test_casual))
# print(len(casual_mean_list))
# casual_mean = np.mean(y_test_casual - casual_mean_list)
# casual_mean_adjust = [x + (casual_mean / 2) for x in casual_mean_list]
# scores = get_scores("Casual avg with mean diff " + str(casual_mean), y_tests['casual'].tolist(), casual_mean_adjust)
# with open('log.txt', append_write) as file:
#     file.write(scores)
# print(scores)
#
# summed_pred = pd.Series(casual_mean_adjust, name='casual results') + mean_pred['registered']
# scores = get_scores("summed with casual adjust only", y_test, summed_pred)
# with open('log.txt', append_write) as file:
#     file.write(scores)
# print(scores)
#
# # adjusting registered
#
# registered_mean_list = mean_pred['registered']['avg'].tolist()
# registered_mean = mean(y_tests['registered'] - registered_mean_list)
# registered_mean_adjust = [x + (registered_mean / 2) for x in bring_to_zero(registered_mean_list)]
# scores = get_scores("registered avg with mean diff " + str(registered_mean), y_tests['registered'],
#                     registered_mean_adjust)
# with open('log.txt', append_write) as file:
#     file.write(scores)
# print(scores)
#
# summed_pred = pd.Series(casual_mean_adjust, name='casual results') + pd.Series(registered_mean_adjust,
#                                                                                name='registered results')
# scores = get_scores("summed with casual and registered adjust", y_test, summed_pred)
# with open('log.txt', append_write) as file:
#     file.write(scores)
# print(scores)
#
# summed_pred = mean_pred['casual'] + pd.Series(registered_mean_adjust, name='registered results')
# scores = get_scores("summed with registered adjust only", y_test, summed_pred)
# with open('log.txt', append_write) as file:
#     file.write(scores)
# print(scores)
#
