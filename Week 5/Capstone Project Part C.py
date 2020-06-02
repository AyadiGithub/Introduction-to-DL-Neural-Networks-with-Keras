'''
PART C - (Normalized Predictors and 100 Epochs)

Question: How does the mean of the mean squared errors compare to that from Step B?
Answer: Increasing the number of epochs from 50 to 100 significantly decreased then mean of the mean squared error
and stabalized the variance and standard deviation from one random_split state to another. It also increased the computation time.


'''

import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#Lets import our dataset
file = r'https://cocl.us/concrete_data'
df = pd.read_csv(file)
concrete_data = df
concrete_data.head()
concrete_data.shape

#Lets check the dataset for any missing values
concrete_data.describe() #Generating summary metrics
concrete_data.isnull().sum() #Checking for missing values

#We set the 'concrete sample strength' as the y (Target) and X as all the columns that are not 'Strength'
concrete_data_columns = concrete_data.columns
X = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
'''
Normalization is required for Part B of the project

'''
X_normalized = (X - X.mean()) / X.std()
X_normalized.head()
y = concrete_data['Strength']
y.head()


#Saving the number of predictor columns (which is 8)
n_cols = X.shape[1] # number of predictor columns
n_cols


#Defining the Baseline Neural Network
def Baseline_Network():
    model = Sequential()

    #We use the add method to add each Dense layer. We will add 10 neurons in the first layer. We specify the input shape parameter
    #We will use the ReLU activation function
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))

    #Output layer has 1 neuron
    model.add(Dense(1))

    #Next we need to specifiy an optimizer. We will use Adam.
    #For measuring the error, we will use Mean Squared Error
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


#Creating an empty list to fill with Mean Squared Errors
MeanSquaredErrorsList = [] 
for i in range (1, 51): #Creating a loop that will execute 50 times with a random data split each time
     
     #Split random state
     random_state = i
     print("Using random_state split: ", random_state)
     
     #Splitting the dataset using the generated random integer
     X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=random_state)   
     
     #Creating the model
     model = Baseline_Network()
    
     #Lets train the model using X_train and y_train for 50 epochs.
     model.fit(X_train, y_train, epochs=100, verbose=0)
     
     #Model Evaluation
     Evaluation = model.evaluate(X_test, y_test, verbose=0)
     print("Mean Squared Error "+str(i)+" = "+str(Evaluation))
     
     #Predicting Strength for Test set
     y_pred = model.predict(X_test)
     MSE = mean_squared_error(y_test, y_pred)
     MeanSquaredErrorsList.append(MSE)

MeanSquaredErrorsList = np.array(MeanSquaredErrorsList)     
MeanSquaredErrorsList
Mean = np.mean(MeanSquaredErrorsList)   
Standard_deviation = np.std(MeanSquaredErrorsList)
print("The Mean of the Mean Squared Errors = ", Mean)
print("The Standard Deviation of the Mean Squared Errors = ", Standard_deviation)


     


