import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sn
from scipy import stats
from numpy import median
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

filename = 'train.csv'
df_train = pd.read_csv(filename)
print(df_train.head(10))

print(df_train.shape)

df_test = pd.read_csv('test.csv')
print(df_test.head(10))

print (df_test.shape)

df_train = df_train.drop(['atemp', 'casual', 'registered', 'windspeed'], axis=1)
print(df_train.head(10))