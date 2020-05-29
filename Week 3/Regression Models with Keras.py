import keras
import pandas as pd
import numpy as np

#Dense network: a network with all neurons in one layer connected to all neurons in the next layer. 

from keras.models import Sequential #Network we are making consists of linear stack of layers so sequential is best. 
#There are two models in keras. 1) Sequential 2) Model class - used with funcitonal API
from keras.layers import Dense #To build the layers

#Lets import our dataset
file = r'C:\****\Introduction to DL & Neural Networks with Keras\Week 3\concrete_data.csv' #file location
concrete_data = pd.read_csv(file)
concrete_data.head()
concrete_data.shape

#Lets check the dataset for any missing values
concrete_data.describe() #Generating summary metrics
concrete_data.isnull().sum() #Checking for missing values

#We set the 'concrete sample strength' as the target and predictors as all the columns that are not 'Strength'
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
predictors.head()
target = concrete_data['Strength']
target.head()

#Now we Normalize the data by Subtracting the mean and dividing by the Std Deviation
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

#Saving the number of predictors (which is 8)
n_cols = predictors_norm.shape[1] # number of predictors
n_cols

#Lets define a function that creates a Dense NN.
#To create model, we call seqential constructor.

def regression_model():
    model = Sequential()

    #We use the add method to add each Dense layer. We will add 5 neurons in the first layer. We specify the input shape parameter
    #We will use the ReLU activation funciton
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))

    #For the second layer, we will add 5 neurons and use the ReLU funtion again
    model.add(Dense(10, activation='relu'))

    #Output layer has 1 neuron
    model.add(Dense(1))

    #Next we need to specifiy an optimizer. We will use Adam, its more efficient that gradient descent
    #And doesnt require a learning rate. 
    #For measuring the error, we will use Mean Squared Error
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


model = regression_model()


#Next we use the fit method to train the model
#Data set is divided into predictors (inputs) and targets
#We will use predictors_norm (Normalized predictors)
#We use 100 epochs and verbose = 2 (To show the training progress)

model.fit(predictors_norm, target, validation_split=0.3, epochs=500, verbose=2)















