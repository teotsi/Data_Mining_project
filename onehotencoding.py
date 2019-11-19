import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

filename = 'Data_Mining_project/train.csv'
one_hot_data = pd.read_csv(filename)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)

one_hot_data.rename(columns={'weathersit':'weather',
                     'mnth':'month',
                     'hr':'hour',
                     'yr':'year',
                     'hum': 'humidity',
                     'cnt':'count'},inplace=True)

one_hot_data['season'] = one_hot_data.season.astype('category')
one_hot_data['year'] = one_hot_data.year.astype('category')
one_hot_data['month'] = one_hot_data.month.astype('category')
one_hot_data['hour'] = one_hot_data.hour.astype('category')
one_hot_data['holiday'] = one_hot_data.holiday.astype('category')
one_hot_data['weekday'] = one_hot_data.weekday.astype('category')
one_hot_data['workingday'] = one_hot_data.workingday.astype('category')
one_hot_data['weather'] = one_hot_data.weather.astype('category')

X = one_hot_data.loc[:,['weather']]
y = one_hot_data.count
print(y)

# logreg = LogisticRegression(solver='lbfgs')
# cross_val_score(logreg, X, y, cv=5, scoring='accuracy').mean()