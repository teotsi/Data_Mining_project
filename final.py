from sklearn.model_selection import train_test_split

from Reader import *



df = read_data('train.csv')
df_test = read_data('test.csv')

X, y = select_train_columns(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_test['weather_4'] = 0
df_test = select_train_columns(df_test)[0]

nn = sequential_nn_model(X_train,y_train)  # fitting neural network model on X and y
nn_pred = nn.predict(X_test)  # making prediction
nn_pred = [transform_list_item(x) for x in nn_pred]

print_scores("Neural Network", y_test, nn_pred)
