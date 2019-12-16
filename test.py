import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor

# matplotlib inline
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from Reader import *
from tens import sequential_nn_model


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

y = df_train['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Tensorflow Sequential Neural Network
Seq_NN = sequential_nn_model(X_train, y_train)

df_test['weather_4'] = 0
df_test = df_test[[x for x in all_columns if x.startswith(tuple(train_columns))]]
# df_test = df_test.drop(columns=['weather_4'])

test_array=df_test.to_numpy()
predictions=Seq_NN.predict(test_array)

individual_predictions = [transform_list(x) for x in predictions]
for i, y in enumerate(individual_predictions):
    if individual_predictions[i] < 0:
        individual_predictions[i] = 0

# print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, individual_predictions)))
# print('R2:', r2_score(y_test, individual_predictions))

# reg_bag = BaggingRegressor(reg)
# reg_bag.fit(X, y)
# df_test['weather_4'] = 0
# df_test = df_test[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
# y_pred = est.predict(df_test)
# for i, y in enumerate(y_pred):
#     if y_pred[i] < 0:
#         y_pred[i] = 0

submission = pd.DataFrame()
submission['Id'] = range(individual_predictions.shape[0])
submission['Predicted'] = individual_predictions
submission.to_csv("submission.csv", index=False)
