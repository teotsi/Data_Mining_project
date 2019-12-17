from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from Reader import *

# reading data without one-hot encoding, since that's optimal for RF
df = read_data('train.csv', one_hot=False)
df_test = read_data('test.csv', one_hot=False)


X, y = select_train_columns(df)

# Training and test data is created by splitting the main data. 30% of test data is considered
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_test = select_train_columns(df_test)[0]  # getting the correctly encoded parameters

# using Random Forest
rf = RandomForestRegressor(n_jobs=-1, max_depth=25, n_estimators=400, random_state=0)
rf.fit(X, y)
rf_pred = rf.predict(df_test)

# print_scores("Random Forest", y_test, rf_pred)

# reading data again to apply one-hot encoding

df = read_data('train.csv')
df_test = read_data('test.csv')

all_columns = list(df.columns)
# Training and test data is created by splitting the main data. 30% of test data is considered
train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity',
                 'windspeed']
X = df[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
y = df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

nn = sequential_nn_model(X, y)

df_test['weather_4'] = 0
df_test = df_test[[x for x in all_columns if x.startswith(tuple(train_columns))]]

nn_pred = nn.predict(df_test.to_numpy())
# print_scores("Neural Network", y_test, nn_pred)
nn_pred = [transform_list_item(x) for x in nn_pred]

merged_pred = pd.concat([pd.Series(nn_pred, name='pred_nn'), pd.Series(rf_pred, name='pred_rf')], axis=1)
print(merged_pred.head(5))
# getting the mean
mean_pred = pd.DataFrame()
mean_pred['avg'] = merged_pred.mean(axis=1)
print(mean_pred['avg'].head(5))

# print_scores("mean predictions",y_test, mean_pred.to_numpy())
create_submission(mean_pred)
