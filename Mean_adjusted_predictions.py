import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# Load data
import Reader
from Experiment.Reader import print_scores
from Reader import sequential_nn_model, transform_list_item

df_train = Reader.read_data('train.csv', one_hot=False)
df_test = Reader.read_data('test.csv', one_hot=False)

X,y = Reader.select_train_columns(df_train)
df_test = Reader.select_train_columns(df_test)[0]
print(X.dtypes)
print(X.head())
print(X.isnull().sum())
print(y.isnull().sum())
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape)
print(y_train.shape)
# Random Forest
# rf = RandomForestRegressor(n_jobs=-1, random_state=0)
# parameters = {'n_estimators': [200, 400], 'max_depth': [15, 25]}
# rf_cv = GridSearchCV(rf, parameters, cv=5, n_jobs=-1)
# rf_cv.fit(X_train, y_train)
# y_pred = rf_cv.predict(X_test)
rf = RandomForestRegressor(n_jobs=-1, max_depth=75, n_estimators=900, random_state=0)

mlp = MLPRegressor(hidden_layer_sizes=(100, 60, 40, 20), activation='relu', solver='lbfgs', alpha=0.0001, verbose=False,
                   max_iter=400)

stacking = StackingRegressor(estimators=[("mlp", mlp), ("randomForest", rf)],
                             n_jobs=-1)
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
for i, y in enumerate(y_pred):
    if y_pred[i] < 0:
        y_pred[i] = 0
merged_pred=[]
merged_pred.append(pd.Series(y_pred, name='pred_rf' + str(1)))
df_test = Reader.read_data('test.csv')
df_train = Reader.read_data('train.csv')
X,y = Reader.select_train_columns(df_train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

df_test = Reader.select_train_columns(df_test)[0]
df_test['weather_4']=0
# X = df_train.drop(['casual', 'registered', 'cnt', 'atemp', 'windspeed'], axis=1)
# y = df_train['cnt']
nn = sequential_nn_model(X_train, y_train)  # fitting neural network model on X and y
nn_pred = nn.predict(X_test)  # making prediction
nn_pred = [transform_list_item(x) for x in nn_pred]
merged_pred.append(pd.Series(nn_pred, name='nn_pred_rf' + str(1)))
merged_pred_df = pd.concat(merged_pred, axis=1)
print(merged_pred_df.head(5))
# getting the mean
mean_pred = pd.DataFrame()
mean_pred['avg'] = merged_pred_df.mean(axis=1)  # getting the mean average of the columns

print_scores("mean predictions", y_test, mean_pred['avg'].tolist())
# Reader.create_submission(mean_pred['avg'].tolist(), filename='summed_submission.csv')
mean=np.mean(y_test-mean_pred['avg'].tolist())
print(mean)
mean_list=mean_pred['avg'].tolist()
mean_list=[x+(mean/2) for x in  mean_list]
print_scores('adjusted predictions', y_test, mean_list)
print(np.mean(y_test-mean_list))