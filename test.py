import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor

# matplotlib inline
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from Reader import *


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)





filename = 'train.csv'
df_train = read_data(filename)
# print(df_train.head(10))

df_test = read_data('test.csv')  # reading test file
# print(df_test.head(10))

# print(df_train.head(10))
# print(df_train.head(10))
# Training
all_columns = list(df_train.columns)
# Training and test data is created by splitting the main data. 30% of test data is considered
train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity']
X = df_train[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
# print(X)
y = df_train['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = tf.estimator.LinearClassifier(X)
reg.train(X_train, max_steps=100)
result = reg.evaluate(X_test)
print(pd.Series(result))
#print(reg.score(X_test, y_test))

n_batches = 1
est = tf.estimator.BoostedTreesClassifier(X, n_batches_per_layer=n_batches)
est.train(X_train, max_steps=100)
result2=est.evaluate(X_test)
print(pd.Series(result2))

y_pred = reg.predict(X_test)
# print(y_pred)
for i, y in enumerate(y_pred):
    if y_pred[i] < 0:
        y_pred[i] = 0

print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))

# reg_bag = BaggingRegressor(reg)
# reg_bag.fit(X, y)
df_test['weather_4'] = 0
df_test = df_test[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
y_pred = est.predict(df_test)
for i, y in enumerate(y_pred):
    if y_pred[i] < 0:
        y_pred[i] = 0

submission = pd.DataFrame()
submission['Id'] = range(y_pred.shape[0])
submission['Predicted'] = y_pred
submission.to_csv("submission.csv", index=False)
