import numpy as np
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import Reader
from sklearn.ensemble import IsolationForest

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)

#This file is for feature engineering
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#------------------Data Engineering Ideas------------------------
# 1)Drop season column beacuse we a have month column
# 2)Don't remove windspeed. Instead make it a categorical value(e.g. <.4=Not Windy || <.7=Windy || >.7=Windy)
# 3)Same with the rest of the non-categorical features(temp, humidity, etc)
#------------------Data Engineering Ideas------------------------


#Normalizes the given column
def normalizer(column, df):
    x = df[[column]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)


    for i in range(len(x_scaled)):
        if x_scaled[i] <= 0.25:
            x_scaled[i] = int(0)
        elif x_scaled[i] <=0.5:
            x_scaled[i] = int(1)
        elif x_scaled[i] <=0.75:
            x_scaled[i] = int(2)
        else: x_scaled[i] = int(3)
    return x_scaled

train_wind_scaled = normalizer('windspeed', df)
train_humidity_scaled = normalizer('hum', df)

test_wind_scaled = normalizer('windspeed', df_test)
test_humidity_scaled = normalizer('hum', df_test)

df[['windspeed']] = train_wind_scaled
df[['hum']] = train_humidity_scaled

df_test[['windspeed']] = test_wind_scaled
df_test[['hum']] = test_humidity_scaled

df = Reader.read_data(df)
df_test = Reader.read_data(df_test)

#Training
df = df.drop(columns=['weather_4'])
all_columns = list(df.columns)
# Training and test data is created by splitting the main data. 30% of test data is considered
train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
X = df[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
y = df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_test = df_test[[x for x in all_columns if x.startswith(tuple(train_columns))]]

clf = IsolationForest(contamination=0.05)
clf.fit(X)
pred_outlier = clf.predict(X)
print('Number of outliers: ', (len(pred_outlier)-pred_outlier.sum())/2)

for i in range(X.shape[0]):
    if pred_outlier[i]==-1:
        X=X.drop([i])
        y=y.drop([i])


NNmodel = Reader.sequential_nn_model(X, y)

df_test['weather_4'] = 0
df_test = df_test[[x for x in all_columns if x.startswith(tuple(train_columns))]]
test_array = df_test.to_numpy()
predictions = NNmodel.predict(test_array)


individual_predictions = [Reader.transform_list_item(x) for x in predictions]
for i, y in enumerate(individual_predictions):
    if individual_predictions[i] < 0:
        individual_predictions[i] = 0

submission = pd.DataFrame()
submission['Id'] = range(len(individual_predictions))
submission['Predicted'] = individual_predictions
submission.to_csv("submission.csv", index=False)