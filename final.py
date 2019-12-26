from sklearn.model_selection import train_test_split

from Reader import *

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)
df = read_data('train.csv', one_hot=False)
df_test = read_data('test.csv', one_hot=False)
print(df.head())

train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather',
                         'temp', 'humidity',
                         'Count_By_Month_of_Year_avg', 'year_day_cnt_avg',
                         'Month_day_cnt_avg','cnt_per_holiday_by_mnth']

X, y = select_train_columns(df, train_columns=train_columns)
X, y = isolation_forest(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_test = select_train_columns(df_test, train_columns=['season', 'month', 'hour', 'weekday', 'workingday', 'weather',
                                                       'temp', 'humidity', 'holiday'])[0]
# features = df.columns
# selector = SelectKBest(f_classif, k=5)
# df2=selector.fit_transform(df[features], df["count"])
# print(df2)
nn = RandomForestRegressor(max_depth=75, n_estimators=900, random_state=42)
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)
y_pred = bring_to_zero(y_pred)
print(get_scores("NN with all columns", y_test, y_pred))
# create_submission(y_pred,filename="RF.csv")
# scores = -np.log10(selector.pvalues_)
# for i in range(len(features)):
#     print(features[i]+' '+str(scores[i]))
