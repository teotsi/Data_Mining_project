from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from Reader import *

# reading data without one-hot encoding, since that's optimal for RF
df = read_data('train.csv', one_hot=False)
df_test = read_data('test.csv', one_hot=False)

X, y = select_train_columns(df)

# Training and test data is created by splitting the main data. 30% of test data is considered
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_test = select_train_columns(df_test)[0]  # getting the correctly encoded parameters
merged_pred = []

# ----- using Random Forest -----
for i in range(10):
    rf = RandomForestRegressor(n_jobs=-1, max_depth=75, n_estimators=900, random_state=0)
    rf.fit(X,y)
    rf_pred = rf.predict(df_test)
    merged_pred.append(pd.Series(rf_pred, name='pred_rf' + str(i)))

# print_scores("Random Forest", y_test, rf_pred)

# reading data again to apply one-hot encoding

df = read_data('train.csv')
df_test = read_data('test.csv')

X, y = select_train_columns(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # splitting

df_test['weather_4'] = 0
df_test = select_train_columns(df_test)[0]

# ------- using neural network -----
for i in range(15):
    nn = sequential_nn_model(X,y)  # fitting neural network model on X and y
    nn_pred = nn.predict(df_test.to_numpy())  # making prediction
    nn_pred = [transform_list_item(x) for x in nn_pred]
    merged_pred.append(pd.Series(nn_pred, name='pred_nn' + str(i)))
# print_scores("Neural Network", y_test, nn_pred)

# ------- using mlp -------
for i in range(5):
    mlp = MLPRegressor(hidden_layer_sizes=(100, 60, 40, 20), activation='relu', solver='lbfgs', alpha=0.0001,
                       verbose=False,
                       max_iter=400)
    mlp.fit(X,y)
    mlp_pred = mlp.predict(df_test)
    merged_pred.append(pd.Series(mlp_pred, name='pred_mlp' + str(i)))

# merging the results from each method in a single dataframe
merged_pred_df = pd.concat(merged_pred, axis=1)
print(merged_pred_df.head(5))
# getting the mean
mean_pred = pd.DataFrame()
mean_pred['avg'] = merged_pred_df.mean(axis=1)  # getting the mean average of the columns

# print_scores("mean predictions", y_test, mean_pred['avg'].tolist())
create_submission(mean_pred)
#
# xgb_regressor = xgb.XGBRegressor(verbosity=3,)
# xgb_regressor.fit(X,y)
# xgb_pred = xgb_regressor.predict(df_test)
# create_submission(xgb_pred)
