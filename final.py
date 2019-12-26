from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from Reader import *


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)
df = read_data('train.csv', one_hot=False)
df_test = read_data('test.csv', one_hot=False)

train_columns =['season', 'month', 'hour', 'weekday', 'workingday', 'weather',
                                                       'temp', 'humidity', 'holiday',
                                                       'Count_By_Month_of_Year_avg', 'year_day_cnt_avg',
                                                       'Month_day_cnt_avg']



X, y = select_train_columns(df, pred_column='count')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_test = select_train_columns(df_test, train_columns)[0]

#rf = ExtraTreesRegressor(n_jobs=-1, max_depth=75, n_estimators=900, random_state=0)
#gb = GradientBoostingRegressor(loss='huber', alpha=0.01, n_estimators = 800, max_depth=25, warm_start=True)

# gb.fit(X_train, y_train)
# y_pred = gb.predict(X_test)
gb_rf = gradient_boost_with_extra_trees(X, y)
y_pred = gb_rf.predict(df_test)
y_pred = bring_to_zero(y_pred)
#print(get_scores("RF with all columns", y_test, y_pred))
create_submission(y_pred,filename="final.csv")
# scores = -np.log10(selector.pvalues_)
# for i in range(len(features)):
#     print(features[i]+' '+str(scores[i]))
