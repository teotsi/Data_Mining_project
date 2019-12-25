from sklearn.model_selection import train_test_split

from Reader import *

df = read_data('train.csv',one_hot=False)
df_test = read_data('test.csv')

X, y = select_train_columns(df,pred_column='count')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_test['weather_4'] = 0
df_test = select_train_columns(df_test, train_columns='casual')[0]

model = gradient_boost_with_random_forest(X_train, y_train)
y_pred = model.predict(X_test)
print(get_scores("gradient boost and stacking random forest",y_test,bring_to_zero(y_pred)))