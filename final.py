import pandas as pd
import numpy as np

import tensorflow as tf 
import keras 
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.layers.recurrent import LSTM
from tensorflow_core.python.keras.models import Sequential
from keras.models import load_model

import Reader
import os 
from sklearn.ensemble.iforest import IsolationForest
from numpy.lib.shape_base import dstack

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from keras.losses import mean_squared_logarithmic_error


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)

df_train = Reader.read_data('train.csv')
df_test = Reader.read_data('test.csv')

#Training
initial_columns = list(df_train.columns)

#Training and test data is created by splitting the main data. 70% training, 30% testing
train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity']
X = df_train[[x for x in initial_columns if x.startswith(tuple(train_columns))]]
y = df_train['count']

#Outlier Detection
clf=IsolationForest(contamination=0.05)
clf.fit(X)
pred_outlier = clf.predict(X)
print('Number of outliers: ', (len(pred_outlier)-pred_outlier.sum())/2)
print(pred_outlier.shape)
print(X.shape)

for i in range(X.shape[0]):
    if pred_outlier[i]==-1:
        X=X.drop([i])
        y=y.drop([i])


#Spliting the test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)

#Creating the model 
n_members = 5
for i in range(n_members):
	# fit model
	model = Reader.sequential_nn_model(X_train, y_train)
	# save model
	filename = 'model_' + str(i + 1) + '.h5'
	model.save(filename)
	print('>Saved %s' % filename)

def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'model_' + str(i + 1) + '.h5'
		# load model from file
		model = tf.keras.models.load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
	model = MLPRegressor()
	model.fit(stackedX, inputy)
	return model

def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat

# load all models
n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

model = fit_stacked_model(members, X_test, y_test)
yhat = stacked_prediction(members, model, X_test)
acc = mean_squared_logarithmic_error(y_test, yhat)
print('Mean squared log error: %.3f' % acc)




# df_test['weather_4'] = 0
# df_test = df_test[[x for x in initial_columns if x.startswith(tuple(train_columns))]]
# test_array = df_test.to_numpy()
# predictions=NNmodel.predict(test_array)

for i, y in enumerate(yhat):
    if yhat[i] < 0:
        yhat[i] = 0

submission = pd.DataFrame()
submission['Id'] = range(len(yhat))
submission['Predicted'] = yhat
submission.to_csv("submission.csv", index=False)
