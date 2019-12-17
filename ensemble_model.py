from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.linear_model import Lasso, ElasticNetCV
from tensorflow_core import metrics

from tens import sequential_nn_model
from Reader import *
from sklearn.model_selection import train_test_split
import numpy as np


def EnsembleRegressor(regressors,X_Validation):
    EnsemblePred = pd.DataFrame()
    for reg in regressors:
        colname = str(reg)[:4]
        EnsemblePred[colname] = reg.predict(X_Validation)
    EnsemblePred["Ensemble"] = EnsemblePred.apply(lambda x: np.mean(x), axis=1) #Mean scores better than median
    return EnsemblePred

def RMSE_log(true,pred):
    RMSE = metrics.mean_squared_error(np.log(true),np.log(pred))**0.5
    return RMSE


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)


filename = 'train.csv'
df_train = read_data(filename)
df_test = read_data('test.csv')  # reading test file

# Training
df_train = df_train.drop(columns=['weather_4'])
all_columns = list(df_train.columns)
# Training and test data is created by splitting the main data. 30% of test data is considered
train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity']
X = df_train[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
y = df_train['count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_test = df_test[[x for x in all_columns if x.startswith(tuple(train_columns))]]
df_test = df_test.to_numpy()

#ElasticNetCV to find the proper variables for the model
ElNet_model = ElasticNetCV(l1_ratio=np.arange(0.05,0.95,0.05),alphas=np.arange(1,100,10),cv=5, max_iter=100)
ElNet_model.fit(X_train, y_train)
ElNet_pred = ElNet_model.predict(X_test)
bring_to_zero(ElNet_pred)

print_scores('ElasticNet', y_test, ElNet_pred)

Lasso_model = Lasso(alpha=150,max_iter=3000)
Lasso_model.fit(X_train,y_train)
Lasso_pred = Lasso_model.fit(X_test,y_test)
Lasso_pred = Lasso_model.predict(X_test)
bring_to_zero(Lasso_pred)

print_scores('Lasso', y_test, Lasso_pred)

#Tensorflow Neural Network
Seq_NN = sequential_nn_model(X_train,y_train)
NN_pred = Seq_NN.predict(X_test)
individual_NN_pred = [transform_list_item(x) for x in NN_pred]
bring_to_zero(individual_NN_pred)

print_scores('Sequential Neural Network', y_test, individual_NN_pred)

#--Outlier Detection------------------------------------------------------
# clf = IsolationForest(contamination=0.05)
#
# X['temp'] = df_train.temp
# X['humidity'] = df_train.humidity
# clf.fit(X)
# pred_outlier = clf.predict(X)
# print('Number of outliers: ', (len(pred_outlier)-pred_outlier.sum())/2)
#--Outlier Detection------------------------------------------------------

EnsembleReg_pred = EnsembleRegressor([ElNet_model, Lasso_model, Seq_NN], X_test)
type(EnsembleReg_pred)
#print_scores('Ensemble Regressor', y_test, EnsembleReg_pred)
print("***ENSEMBLE REGRESSOR***\nRMSE for test set : {}".format(RMSE_log(pred=EnsembleReg_pred.Ensemble,true=y_test)))