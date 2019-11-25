import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
<<<<<<< HEAD

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
=======
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

filename = "train.csv"
ohd = pd.read_csv(filename)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)

ohd.rename(columns={'weathersit':'weather',
                    'mnth':'month',
                    'hr':'hour',
                    'yr':'year',
                    'hum':'humidity',
                    'cnt':'count'}, inplace=True)

ohd['season'] = ohd.season.astype('category')
ohd['year'] = ohd.year.astype('category')
ohd['month'] = ohd.month.astype('category')
ohd['hour'] = ohd.hour.astype('category')
ohd['holiday'] = ohd.holiday.astype('category')
ohd['weekday'] = ohd.weekday.astype('category')
ohd['workingday'] = ohd.workingday.astype('category')
ohd['weather'] = ohd.weather.astype('category')

le = LabelEncoder()
ohd_2 = ohd.apply(le.fit_transform)

#print(ohd.head(5))

one_hot_encoder = OneHotEncoder(categorical_features=[0,1,2,3,4,5,6,7])   #This command separates the first 8 columns into binary columns
one_hot_encoder.fit(ohd_2)

#print(ohd_2.head(10))

oneHotLabels = one_hot_encoder.fit_transform(ohd_2).toarray()   #This is why this array has 18 columns instead of 15(The first 4 are the binary columns)
print(oneHotLabels[:10])
#print(ohd.columns)
#print(ohd.head(10))

>>>>>>> 484eaecff2045f84f11aa58e15fb8bb9e278ef06
