#The MNIST database, short for Modified National Institute of Standards and Technology database, 
#is a large database of handwritten digits that is commonly used for training various image processing systems. 
#The database is also widely used for training and testing in the field of machine learning.
#The MNIST database contains 60,000 training images and 10,000 testing images 
#of digits written by high school students and employees of the United States Census Bureau.

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

#We import pyplot from Matplotlib to view the images
import matplotlib.pyplot as plt

#We import the MNIST dataset
from keras.datasets import mnist

#Lets read the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Lets confirm the no. of images in the dataset. It should be 60,000 in X_train and 10,000 in X_test
X_train.shape 
X_test.shape

#Lets visualize the first img
plt.imshow(X_train[0])

#We need to flatten the images into one-dimensional vectors, each of size 1 x (28 x 28) = 1 x 784. So we can use it in a Conventional NN.
num_pixels = X_train.shape[1] * X_train.shape[2] #28 x 28
num_pixels
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') #Flattening training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') #Flattening test images

#New shapes should be 60000,784 and 10000,784 
X_train.shape
X_test.shape

#Some values for pixels are ranging from 0 to 255. So set needs Normalization
X_train = X_train / 255
X_test = X_test / 255

#One Hot Encoding Outputs to divide the target variable into categories for Classificaiton
#We use to_categorical to accomplish this
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#We assign the number of categories
num_classes = y_test.shape[1]
print(num_classes)

#Now we define the function to build the NN.
#define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    #Compile model using adam optimizer and categorical crossentropy to calculate the error. 
    #Metrics will use built-in 'Accuracy' metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = classification_model()

#Use X_train and y_train for training and use 0.1 of it for validation
model.fit(X_train, y_train, validation_split=0.1, epochs=10, verbose=2)

#Evaluating the model using the X_test and y_test that the model has not seen 
scores = model.evaluate(X_test, y_test, verbose=0)

#Lets show the accuracy and error of our model
print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))      



#Lets save our trained model
model.save(r'****\Introduction to DL & Neural Networks with Keras\Week 3\classification_model.h5')


#To load the model.
from keras.models import load_model
pretrained_model = r'*****\Introduction to DL & Neural Networks with Keras\Week 3\classification_model.h5'
load_model(pretrained_model)
