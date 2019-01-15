import numpy 
import pandas
import csv as csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


train=[]
train_formation=[]
train_bandgap=[]
test=[]   
test_formation=[]
test_bandgap=[]      #Array Definition

path1 =  r'..\input\train.csv'     #Address Definition
path2 =  r'..\input\test.csv'

with open(path1, 'r') as f1:    #Open File as read by 'r'
    reader = csv.reader(f1)     
    next(reader, None)          #Skip header because file header is not needed
    for row in reader:          #fill array by file info by for loop
        train.append(row)
    train = numpy.array(train)       	
	
with open(path2, 'r') as f2:
    reader2 = csv.reader(f2)
    next(reader2, None)  
    for row2 in  reader2:
        test.append(row2)
    test = numpy.array(test)
 
train = numpy.delete(train,[0],1)  #delete first column
 

#predicting formation_energy_ev_natom


train_formation = train 
train_formation = numpy.delete(train,[13],1)  #delete last column

test_formation = test
test_formation = numpy.delete(test,[0],1)  #delete first column 

X = train_formation[:,0:11]
Y = train_formation[:,11]


x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.5, random_state=101)

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
	
#train model
modelR = baseline_model()
modelR.fit(x_valid, y_train, batch_size = 5, epochs=10)

#evaluate
modelR.evaluate(x_valid, y_valid, batch_size=5)
	
#predict
classes = modelR.predict(test_formation)
	
#print output to file
path3 =  r'..\output\result_keras.csv'

with open(path3, 'w',  newline='') as f3, open(path2, 'r') as f4: # write output and otherr column from test
    forest_Csv = csv.writer(f3)
	#forest_Csv.writerow(["id", "formation_energy", "spacegroup", "number_of_total_atoms", "percent_atom_al", "percent_atom_al", "percent_atom_in", "lattice_vector_1_ang", "lattice_vector_2_ang", "lattice_vector_3_ang", "lattice_angle_alpha_degree", "lattice_angle_beta_degree", "lattice_angle_gamma_degree"])       
    test_file_object = csv.reader(f4)
    next(test_file_object, None)
    i = 0
    for row in  test_file_object:
        row.insert(1,classes[i].astype(numpy.float16))
        forest_Csv.writerow(row)
        i += 1	
	
	
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)

#calculate MSE
kfold = KFold(n_splits=2, random_state=seed)
results = cross_val_score(estimator, x_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))



# evaluate model with standardized dataset : To even the scales of input attribute
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=100, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=100, random_state=seed)
results = cross_val_score(pipeline, x_train, y_train, verbose = 1)

#train
pipeline.fit(X, Y)

#predict
predicted_formation = pipeline.predict(test_formation)

print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#print output to file
path3 =  r'..\output\Standardized_Result_Formation.csv'

with open(path3, 'w',  newline='') as f3, open(path2, 'r') as f4: # write output and otherr column from test
    forest_Csv = csv.writer(f3)
    #forest_Csv.writerow(["id", "formation_energy", "spacegroup", "number_of_total_atoms", "percent_atom_al", "percent_atom_al", "percent_atom_in", "lattice_vector_1_ang", "lattice_vector_2_ang", "lattice_vector_3_ang", "lattice_angle_alpha_degree", "lattice_angle_beta_degree", "lattice_angle_gamma_degree"])
    test_file_object = csv.reader(f4)
    next(test_file_object, None)
    i = 0
    for row in  test_file_object:
        row.insert(1,predicted_formation[i].astype(numpy.float16))
        forest_Csv.writerow(row)
        i += 1	
		


		
#predicting bandgap_energy_ev : Done only for standardized data set 

train_bandgap = train 
train_bandgap = numpy.delete(train,[12],1)  #delete formation_energy column


test_bandgap = test
test_bandgap = numpy.delete(test,[0],1)  #delete first column 

X = train_bandgap[:,0:11]
Y = train_bandgap[:,11]


x_train_B, x_valid_B, y_train_B, y_valid_B = train_test_split(X, Y, test_size=0.2, random_state=101)

# evaluate model with standardized dataset : To even the scales of input attribute
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=100, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=100, random_state=seed)
results = cross_val_score(pipeline, x_train_B, y_train_B, verbose = 1)

#train
pipeline.fit(X, Y)

#predict
predicted_test = pipeline.predict(test_bandgap)

print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#print output to file
path3 =  r'..\output\Standardized_Result_Bandgap.csv'

with open(path3, 'w',  newline='') as f3, open(path2, 'r') as f4: # write output and otherr column from test
    forest_Csv = csv.writer(f3)
    #forest_Csv.writerow(["id", "formation_energy", "spacegroup", "number_of_total_atoms", "percent_atom_al", "percent_atom_al", "percent_atom_in", "lattice_vector_1_ang", "lattice_vector_2_ang", "lattice_vector_3_ang", "lattice_angle_alpha_degree", "lattice_angle_beta_degree", "lattice_angle_gamma_degree"])
    test_file_object = csv.reader(f4)
    next(test_file_object, None)
    i = 0
    for row in  test_file_object:
        row.insert(1,predicted_test[i].astype(numpy.float16))
        forest_Csv.writerow(row)
        i += 1	
