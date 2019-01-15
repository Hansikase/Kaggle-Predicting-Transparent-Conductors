
# coding: utf-8

# In[38]:


import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats.mstats import zscore
from sklearn import preprocessing

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from patsy import dmatrices

import seaborn as sns

import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet, ElasticNetCV


# In[ ]:


#Read data 


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

x = train.iloc[:,1:12]
y = train.iloc[:,12]


# In[ ]:


train1 = pd.DataFrame()
train1 = train.iloc[:,1:12]

data = preprocessing.scale(train1)
scaled = preprocessing.normalize(data)
#print (x.size)
#print (train.iloc[0,1:12])



standardized_X = pd.DataFrame(data=data[0:,0:],    # values
              index=train1.index,    # 1st column as index
              columns=train1.columns)


# In[ ]:


sm.OLS(zscore(y), zscore(standardized_X)).fit().summary()


# In[ ]:


cor = train.corr()
plt.figure(figsize=(12,8))
sns.heatmap(cor,cmap='Set1',annot=True)


# In[ ]:


#Calculate VIF factor


# In[ ]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(standardized_X.values, i) for i in range(standardized_X.shape[1])]
vif["features"] = standardized_X.columns
vif.round(1)




# def baseline_model():
	# regressor = Sequential()
	# regressor.add(Dense(units = 128, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 8))
	# regressor.add(Dropout(0.1))
	# regressor.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'random_uniform'))
	# regressor.add(Dropout(0.1))
	# regressor.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'random_uniform'))

	# #compile ANN
	# regressor.compile(optimizer = 'adam', loss = 'mse', metrics =['accuracy'])
	# return regressor

#binary_crossentropy
# In[ ]:


# estimators = []
# estimators.append(('standardize', StandardScaler()))
# #estimators.append(('normalize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=1, validation_split=0.1)))
# model = Pipeline(estimators)


# Remove columns
#x = x.drop(['percent_atom_al', 'percent_atom_ga', 'percent_atom_in'], axis=1)
x = x.drop(['percent_atom_al'], axis=1)


# In[ ]:


x_train, x_valid, y_train, y_valid = train_test_split(x.iloc[:,:9], y, test_size=0.2, random_state=101)


#calculate alpha and l1_ratio using cross validation

regr = ElasticNetCV(cv=5, random_state=0)
regr.fit(x_train, y_train)

ElasticNetCV(alphas=None, copy_X=True, cv=5, eps=0.001, fit_intercept=True,
       l1_ratio=0.5, max_iter=1000, n_alphas=100, n_jobs=1,
       normalize=False, positive=False, precompute='auto', random_state=0,
       selection='cyclic', tol=0.0001, verbose=0)
	   
print("elastic net alpha : ", regr.alpha_) 
print("elastic net  l1 ratio : ", regr.intercept_) 

alpha_ = regr.alpha_
ratio_ = regr.intercept_


# In[ ]:


# Min and max degree of polynomials features to consider
degree_min = 1
degree_max = 4

# Make a pipeline model with polynomial transformation and ElasticNet regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
for degree in range(degree_min,degree_max+1):
	model = make_pipeline(StandardScaler(), PolynomialFeatures(degree, interaction_only=False), ElasticNet(alpha=alpha_, l1_ratio=ratio_, max_iter=10000, normalize=False, positive=False, precompute=False,
      random_state=0, selection='cyclic', tol=0.0001, warm_start=False, fit_intercept=True))
	model.fit(x_train,y_train)
	print ("trained degree: ",degree)
	valid_pred_f = np.array(model.predict(x_valid))
	#final_pred_f = np.array(model.predict(test))
	RMSE=np.sqrt(np.sum(np.square(valid_pred_f-y_valid)))
	test_score = model.score(x_valid,y_valid)

#model.fit(x_train,y_train)


#valid_pred_f = np.array(model.predict(x_valid))
#final_pred_f = np.array(model.predict(test))

# In[ ]:

# RMSE=np.sqrt(np.sum(np.square(valid_pred_f-y_valid)))
# test_score = model.score(x_valid,y_valid)
# results = cross_val_score(model, x_valid, y_valid, verbose = 1)


# In[ ]:

#results = cross_val_score(model, x_al, y, verbose = 1)
#print("Standardized: %.10f (%.10f) MSE  " % (results.mean(), results.std()))

# classifier = model.best_estimator_.named_steps['mlp']
#print ("cross val score : ", results)

print ("RMSE: ",RMSE)
print ("test score: ",test_score)

# The coefficients
# model.named_steps_[2].coef_
# print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.10f"
      % mean_squared_error(y_valid, valid_pred_f))
# Explained variance score: 1 is perfect prediction
print('Variance/R^2 score: %.10f' % r2_score(y_valid, valid_pred_f))


#plt.plot(history.history['mean_squared_error'])
#plt.show()
