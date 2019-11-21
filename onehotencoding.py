import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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

