import statsmodels
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

class StationarityTests:
    def __init__(self, significance=.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None

    def ADF_Stationarity_Test(self, timeseries, printResults=True):
        # Dickey-Fuller test:
        adfTest = adfuller(timeseries, autolag='AIC')

        self.pValue = adfTest[1]

        if (self.pValue < self.SignificanceLevel):
            self.isStationary = True
        else:
            self.isStationary = False

        if printResults:
            dfResults = pd.Series(adfTest[0:4],
                                  index=['ADF Test Statistic', 'P-Value', '# Lags Used', '# Observations Used'])
            # Add Critical Values
            for key, value in adfTest[4].items():
                dfResults['Critical Value (%s)' % key] = value
            print('Augmented Dickey-Fuller Test Results:')
            print(dfResults)

df = pd.read_csv('train.csv')
df=df[['cnt']]
dfar=df.values
ar=np.delete(dfar,[1,2],axis=1)
ar=ar.ravel()
sTest = StationarityTests()
sTest.ADF_Stationarity_Test(ar, printResults = True)
print("Is the time series stationary? {0}".format(sTest.isStationary))