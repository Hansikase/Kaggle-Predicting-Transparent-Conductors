import numpy as np 
import pandas as pd
import statsmodels.api as sm
from scipy.stats.mstats import zscore
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split

# Import function to automatically create polynomial features! 
from sklearn.preprocessing import PolynomialFeatures
# Import Linear Regression and a regularized regression function
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
# Finally, import function to make a machine learning pipeline
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import ElasticNet

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.wrappers.scikit_learn import KerasRegressor


train = pd.read_csv('D:/New folder/Kagg;e/TestPro/input/train.csv')
testf = pd.read_csv('D:/New folder/Kagg;e/TestPro/input/test.csv')

xT = train.iloc[:,1:12]
yT = train.iloc[:,12]

print (sm.OLS(zscore(yT), zscore(xT)).fit().summary())

#only consider atomic percentages since the standard coefficient is high for them compared to the rest of the features.
x = train.iloc[:,2:5]
y = train.iloc[:,12]

test = testf.iloc[:,3:6]

print (y)
def baseline_model():
	regressor = Sequential()
	regressor.add(Dense(units = 2048, activation = 'relu', kernel_initializer = 'normal',input_dim = 3))
	regressor.add(Dropout(0.1))
	regressor.add(Dense(units = 1024, activation = 'relu', kernel_initializer = 'normal'))
	regressor.add(Dropout(0.1))
	regressor.add(Dense(units = 512, activation = 'relu', kernel_initializer = 'normal'))
	regressor.add(Dropout(0.1))
	regressor.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'normal'))
	regressor.add(Dropout(0.1))
	regressor.add(Dense(units = 1, activation = 'relu', kernel_initializer = 'normal'))

	#compile ANN
	regressor.compile(optimizer = 'adam', loss = 'mse', metrics =['accuracy'])
	return regressor

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=101)

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=50, verbose=1, validation_split=0.1)))

model = Pipeline(estimators)
model.fit(x_train,y_train)
valid_pred_f = np.array(model.predict(x_valid))
final_pred_f = np.array(model.predict(test))

RMSE=np.sqrt(np.sum(np.square(valid_pred_f-y_valid)))
test_score = model.score(x_valid,y_valid)
results = cross_val_score(model, x_valid, y_valid, verbose = 1)

print("Standardized: %.10f (%.10f) MSE  " % (results.mean(), results.std()))
	
print ("RMSE: ",RMSE)
print ("test score: ",test_score)
print("Mean squared error: %.10f"
      % mean_squared_error(y_valid, valid_pred_f))
# Explained variance score: 1 is perfect prediction
print('Variance/R^2 score: %.10f' % r2_score(y_valid, valid_pred_f))


#bandgap predictionS

traing = pd.read_csv('D:/New folder/Kagg;e/TestPro/input/train.csv')

x = traing.iloc[:,2:5]
y = traing.iloc[:,13]

test = testf.iloc[:,3:6]

print (y)
def baseline_model():
	regressor = Sequential()
	regressor.add(Dense(units = 2048, activation = 'relu', kernel_initializer = 'normal',input_dim = 3))
	regressor.add(Dropout(0.1))
	regressor.add(Dense(units = 1024, activation = 'relu', kernel_initializer = 'normal'))
	regressor.add(Dropout(0.1))
	regressor.add(Dense(units = 512, activation = 'relu', kernel_initializer = 'normal'))
	regressor.add(Dropout(0.1))
	regressor.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'normal'))
	regressor.add(Dropout(0.1))
	regressor.add(Dense(units = 1, activation = 'relu', kernel_initializer = 'normal'))

	#compile ANN
	regressor.compile(optimizer = 'adam', loss = 'mse', metrics =['accuracy'])
	return regressor
	
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=101)

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=50, verbose=1, validation_split=0.1)))

model = Pipeline(estimators)
model.fit(x_train,y_train)
#test_pred = np.array(model.validate(x_valid))
valid_pred_g = np.array(model.predict(x_valid))
final_pred_g = np.array(model.predict(test))

RMSE=np.sqrt(np.sum(np.square(valid_pred_g-y_valid)))
test_score = model.score(x_valid,y_valid)
results = cross_val_score(model, x_valid, y_valid, verbose = 1)

print("Standardized: %.10f (%.10f) MSE  " % (results.mean(), results.std()))
	
print ("RMSE: ",RMSE)
print ("test score: ",test_score)

# The mean squared error
print("Mean squared error: %.10f"
      % mean_squared_error(y_valid, valid_pred_g))
# Explained variance score: 1 is perfect prediction
print('Variance/R^2 score: %.10f' % r2_score(y_valid, valid_pred_g))

xgb = pd.DataFrame()
xgb['id'] = testf['id']
xgb['formation_energy_ev_natom'] = final_pred_f
xgb['bandgap_energy_ev'] = valid_pred_g
xgb.to_csv("D:/New folder/Kagg;e/TestPro/output/final.csv", index=False)