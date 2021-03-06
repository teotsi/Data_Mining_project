import os
from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from Reader import *

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
        print('gbrf')
        done = False
        while not done:
            try:
                casual_model = gradient_boost_with_extra_trees(X, y)
                done = True
            except ValueError:
                print("Caught ValueError")

        casual_y_pred = casual_model.predict(df_test)
        # log_scores(get_scores("gradient rf boost for casual", y_test, bring_to_zero(casual_y_pred)))
        for i in range(6):
            merged_pred[column].append(pd.Series(casual_y_pred, name='pred_gbrf' + str(i)))

    # ----- using Random Forest -----
    if column =='casual':
        REPETITIONS = 15
    else:
        REPETITIONS = 10
    rf = ExtraTreesRegressor(n_jobs=-1, max_depth=75, n_estimators=900, random_state=0)
    rf.fit(X, y)
    rf_pred = rf.predict(df_test)
    rf_pred = bring_to_zero(rf_pred)
    for i in range(REPETITIONS):
        merged_pred[column].append(pd.Series(rf_pred, name='pred_rf' + str(i)))
    print(merged_pred[column])
    # log_scores(get_scores(column + " RF", merged_pred[column][0], y_test))

    df = read_data('train.csv', extra_csv=column + '.csv')
    df_test = read_data('test.csv', extra_csv=column + '.csv')

    X, y = select_train_columns(df, pred_column=column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # splitting

    df_test['weather_4'] = 0
    df_test = select_train_columns(df_test, pred_column=column)[0]

    # ------- using neural network -----
    i = 0
    if column == 'casual':
        reps = 13
    elif column == 'registered':
        reps = 20
    else:
        reps = 13
    while i < reps:
        nn = sequential_nn_model(X, y)  # fitting neural network model on X and y
        nn_pred = nn.predict(df_test)  # making prediction
        nn_pred = [transform_list_item(x) for x in nn_pred]
        print(nn_pred[0:5])
        if all(pred == 0.0 for pred in nn_pred):
            print("fixing nn")
            continue
        merged_pred[column].append(pd.Series(nn_pred, name='pred_nn' + str(i)))
        i += 1
    # log_scores(get_scores(column + " Neural Network", y_test, nn_pred))

    # ------- using mlp -------
    # for i in range(0):
    #     mlp = MLPRegressor(hidden_layer_sizes=(100, 60, 40, 20), activation='relu', solver='lbfgs', alpha=0.0001,
    #                        verbose=False,
    #                        max_iter=400)
    #     mlp.fit(X_train, y_train)
    #     mlp_pred = mlp.predict(X_test)
    #     mlp_pred = bring_to_zero(mlp_pred)
    #     merged_pred[column].append(pd.Series(mlp_pred, name='pred_mlp' + str(i)))

    # merging the results from each method in a single dataframe
    merged_pred_df[column] = pd.concat(merged_pred[column], axis=1)
    print(merged_pred_df[column].head())
    # getting the mean
    mean_pred[column] = pd.DataFrame()
    mean_pred[column]['avg'] = merged_pred_df[column].mean(axis=1)  # getting the mean average of the columns

    # scores = get_scores(column + " AVG prediction", y_test, bring_to_zero(mean_pred[column]['avg'].tolist()))
    # with open('log.txt', append_write) as file:
    #     file.write(scores)
    # print(scores)
    # mean = np.mean(y_test - mean_pred[column]['avg'].tolist())
    # print(mean)
    # mean_list = mean_pred[column]['avg'].tolist()
    # mean_list = [x + (mean / 2) for x in mean_list]
    # mean_list=bring_to_zero(mean_list)
    # scores = get_scores('adjusted predictions for '+ column, y_test, mean_list)
    # with open('log.txt', append_write) as file:
    #     file.write(scores)
    # print(scores)

# now we are going to merge casual and registered predictions
summed_pred = mean_pred['casual'] + mean_pred['registered']

# log_scores(get_scores("Summed prediction", y_test, summed_pred))
summed_pred_array = summed_pred.to_numpy()
# create_submission(summed_pred_array, filename='sum.csv')
floor_array = deepcopy(summed_pred_array)
floor_array = np.floor(floor_array)
# create_submission(summed_pred_array, filename='sum_floor.csv')

# log_scores(get_scores('Floor summed predictions', y_test, floor_array))
create_submission(floor_array, filename='LastHope.csv')


# total_df = pd.DataFrame({'summed': summed_pred_array[:,0], 'floored': floor_array[:,0]})
# total_df['avg'] = total_df.mean(numeric_only=True, axis=1)
# log_scores(get_scores("mean_summed", y_test, bring_to_zero(total_df['avg'].tolist())))
# summed_pred = mean_pred['registered']['avg'] + merged_pred['test']
# log_scores(get_scores('summed predictions with descent', y_test, summed_pred))

# create_submission(summed_pred, filename='gradient_sum_full.csv')

# summed_pred = np.floor(summed_pred)
# scores = get_scores('Floor summed predictions with descent', y_test, summed_pred)
# with open('log.txt', append_write) as file:
#     file.write(scores)
# print(scores)
# create_submission(summed_pred, filename='gradient_sum_floor.csv')

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
