import numpy as np 
import pandas as pd
import statsmodels.api as sm
from scipy.stats.mstats import zscore
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Import function to automatically create polynomial features! 
from sklearn.preprocessing import PolynomialFeatures
# Import Linear Regression and a regularized regression function
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
# Finally, import function to make a machine learning pipeline
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import ElasticNet


train = pd.read_csv('../input/train.csv')
testf = pd.read_csv('../input/test.csv')

df = train.drop(train.columns[0],axis=1)  #delete first column
df = train.drop('bandgap_energy_ev', axis=1)  #delete last column

x = train.iloc[:,0:12]
y = train.iloc[:,12]

test = testf.iloc[:,3:6]

print (sm.OLS(zscore(y), zscore(x)).fit().summary())

#selectatomic percentages since they have the highest coef compared to the rest of the features
x_percent = x.iloc[:,2:5]

#Feature extraction to remove correlation between the features
pca = decomposition.PCA(n_components=3)
pca.fit(x_percent)
x1 = pca.transform(x_percent)

x_train, x_valid, y_train, y_valid = train_test_split(x_percent, y, test_size=0.2, random_state=101)

# Min and max degree of polynomials features to consider
degree_min = 2
degree_max = 50

# Make a pipeline model with polynomial transformation and ElasticNet regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
for degree in range(degree_min,degree_max+1):
	model = make_pipeline(StandardScaler(), PolynomialFeatures(degree, interaction_only=False), ElasticNet(alpha=300, l1_ratio=0.5, max_iter=1000000, normalize=False, positive=False, precompute=False,
      random_state=0, selection='cyclic', tol=0.0001, warm_start=False, fit_intercept=True))
	model.fit(x_train,y_train)
	print ("trained degree: ",degree)
	test_pred = np.array(model.predict(x_valid))
	final_pred_f = np.array(model.predict(test))
	RMSE=np.sqrt(np.sum(np.square(test_pred-y_valid)))
	test_score = model.score(x_valid,y_valid)
	
results = cross_val_score(model, x_valid, y_valid, verbose = 1)
print("Standardized: %.10f (%.10f) MSE  " % (results.mean(), results.std()))
	
print ("RMSE: ",RMSE)
print ("test score: ",test_score)

# The mean squared error
print("Mean squared error: %.10f"
      % mean_squared_error(y_valid, test_pred))
	  
# Explained variance score: 1 is perfect prediction
print('Variance/R^2 score: %.10f' % r2_score(y_valid, test_pred))

#plot results
x_scat = x_valid.iloc[:,0] #take only one feature

lw = 2
plt.scatter(x_scat, y_valid, color='darkorange', label='data')
plt.plot(x_scat, test_pred, color='navy', lw=lw, label='Pred')
plt.xlabel('data')
plt.ylabel('target')
plt.title('ElasticNet Regression')
plt.legend()
plt.show()

