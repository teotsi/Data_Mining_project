import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from Reader import *
from sklearn.ensemble import IsolationForest, VotingRegressor, StackingRegressor, AdaBoostRegressor, \
    RandomForestRegressor, BaggingRegressor

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)

# This file is for feature engineering
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


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


train_wind_scaled = normalizer('windspeed', df)
train_humidity_scaled = normalizer('hum', df)

test_wind_scaled = normalizer('windspeed', df_test)
test_humidity_scaled = normalizer('hum', df_test)

df[['windspeed']] = train_wind_scaled
df[['hum']] = train_humidity_scaled

df_test[['windspeed']] = test_wind_scaled
df_test[['hum']] = test_humidity_scaled

df = read_data(df, is_dataframe=True, one_hot=False)
df_test = read_data(df_test, is_dataframe=True, one_hot=False)

# Training
# df = df.drop(columns=['weather_4'])
all_columns = list(df.columns)
# Training and test data is created by splitting the main data. 30% of test data is considered
train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity',
                 'windspeed']
X = df[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
y = df['casual']
z = df['registered']
c = df['count']

X_train, X_test, y_train, y_test, z_train, z_test, c_train, c_test = train_test_split(X, y, z, c, test_size=0.3, random_state=42)

df_test = df_test[[x for x in all_columns if x.startswith(tuple(train_columns))]]

clf = IsolationForest(contamination=0.05)
clf.fit(X)
pred_outlier = clf.predict(X)
print('Number of outliers: ', (len(pred_outlier) - pred_outlier.sum()) / 2)

for i in range(X.shape[0]):
    if pred_outlier[i] == -1:
        X = X.drop([i])
        y = y.drop([i])
        z = z.drop([i])

# NNmodel = Reader.sequential_nn_model(X, y)

# ------------------ Predict the casuals ---------------------------------
# knn = KNeighborsRegressor(n_jobs=-1, n_neighbors=2, weights='distance', p=1)
dt = DecisionTreeRegressor(random_state=0)
mlp = MLPRegressor(hidden_layer_sizes=(100, 60, 40, 20), activation='relu', solver='lbfgs', alpha=0.0001, verbose=False,
                   max_iter=400)
rf = RandomForestRegressor(n_jobs=-1, max_depth=25, n_estimators=900, random_state=0)
# adaknn = AdaBoostRegressor(base_estimator=knn, random_state=0, n_estimators=9)
bagdt = BaggingRegressor(base_estimator=dt, n_estimators=300, random_state=0)
# rf.fit(X_train,y_train)
# pred=rf.predict(X_test)
# -------------------- Stacking voting -----------------------------
stacking = StackingRegressor(estimators=[('bagdt', bagdt), ("mlp", mlp), ("randomForest", rf)],
                             n_jobs=-1)
stacking.fit(X, y)
y_pred_stacking = stacking.predict(df_test)
print(y_pred_stacking)

# ------------------ Predict the registered ones -------------------------
# knn = KNeighborsRegressor(n_jobs=-1, n_neighbors=2, weights='distance', p=1)
dt = DecisionTreeRegressor(random_state=0)
mlp = MLPRegressor(hidden_layer_sizes=(100, 60, 40, 20), activation='relu', solver='lbfgs', alpha=0.0001, verbose=False,
                   max_iter=400)
rf = RandomForestRegressor(n_jobs=-1, max_depth=25, n_estimators=900, random_state=0)
# adaknn = AdaBoostRegressor(base_estimator=knn, random_state=0, n_estimators=9)
bagdt = BaggingRegressor(base_estimator=dt, n_estimators=300, random_state=0)
# -------------------- Stacking voting -----------------------------
stacking = StackingRegressor(estimators=[('bagdt', bagdt), ("mlp", mlp), ("randomForest", rf)],
                             n_jobs=-1)
stacking.fit(X, z)
z_pred_stacking = stacking.predict(df_test)

# score function
def rmsle_score(y_true, y_pred):
    for i, y in enumerate(y_pred):
        if y_pred[i] < 0:
            y_pred[i] = 0
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# print('Stacking RMSLE score for casual:', rmsle_score(y_test, y_pred_stacking))
# print('Stacking RMSLE score for registered:', rmsle_score(z_test, z_pred_stacking))
#
# print('Stacking RMSLE score for count by adding casual and registered:', rmsle_score(c_test, np.add(z_pred_stacking,y_pred_stacking)))

create_submission(np.add(z_pred_stacking,y_pred_stacking))