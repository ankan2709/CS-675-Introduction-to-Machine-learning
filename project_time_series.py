'''
Name: Ankan Dash
CS 675 Introduction to Machine Learning
NJIT
Project 3, Time Series predictions
'''

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd

data = sys.argv[1]
df = pd.read_csv(data)


# Taking the 51st column as the testing labels
labels = df['W51'] 

# dropping the Product Code and the 51st column
df = df.drop(['Product_Code', 'W51'], axis = 1)

# dropping the other columns that are not relevant 
df = df.drop(df.loc[:,'MIN':"Normalized 51"], axis = 1)


# time for the X data and converting in to a numpy array 
time = np.array(range(0,51,1))
time = time.reshape(-1,1)


# testing time point numpy array
test = np.array(52)
test = test.reshape(-1, 1)


'''
TRAINING THE MODEL AND MAKING PREDICTIOINS

To fit the data and make predictions I am using the MLP Regressor 
module from Sklearn. 

'''

from sklearn.neural_network import MLPRegressor

clf = MLPRegressor(solver='adam', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

mlp_predictions = []
for i in range(len(df)):
  clf.fit(time, df.loc[i])
  mlp_predictions.append(clf.predict(test).item())


from sklearn.metrics import mean_squared_error
#print('MSE for LinearRegression  =  ',mean_squared_error(lr_predictions,labels))
for i in range(len(labels)):
	print(round(mlp_predictions[i],3))
print('\n')
print('Mean Squared Error (MSE)  =  ',mean_squared_error(mlp_predictions,labels))


