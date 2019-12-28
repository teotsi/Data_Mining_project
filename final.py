from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from Reader import *
from sklearn.linear_model import LinearRegression, LogisticRegression


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)
df = read_data('train.csv', one_hot=False)
df_test = read_data('test.csv', one_hot=False)

training_columns =['season', 'month', 'hour', 'weekday', 'workingday', 'weather',
                                                       'temp', 'humidity', 'holiday'
                                                       'Count_By_Month_of_Year_avg', 'year_day_cnt_avg',
                                                       'Month_day_cnt_avg', 'Avg_casual_by_Weekday_by_Weather', 'windspeed','Avg_casual_on_Workday','cnt_per_holiday_by_mnth']



X, y = select_train_columns(df, train_columns=training_columns, pred_column='count')
#X, y = isolation_forest(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_test = select_train_columns(df_test, train_columns=training_columns)[0]

#rf = ExtraTreesRegressor(n_jobs=-1, max_depth=120, n_estimators=100, random_state=42, verbose=3)
#gb = GradientBoostingRegressor(loss='huber', alpha=0.01, n_estimators = 800, max_depth=25, warm_start=True)
#bg = HistGradientBoostingRegressor(loss='least_squares', max_depth=150)


#rf.fit(X_train, y_train)
#y_pred = rf.predict(X_test)
gb_rf = gradient_boost_with_extra_trees(X_train, y_train)
y_pred = gb_rf.predict(X_test)
y_pred = bring_to_zero(y_pred)
print(get_scores("RF with all columns", y_test, y_pred))
#create_submission(y_pred,filename="final.csv")
# scores = -np.log10(selector.pvalues_)
# for i in range(len(features)):
#     print(features[i]+' '+str(scores[i]))
