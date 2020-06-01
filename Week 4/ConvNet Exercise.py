import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

#Packages for Convolutional Neural Networks
from keras.layers.convolutional import Conv2D #To add Convolutional layers
from keras.layers.convolutional import MaxPooling2D #To add Pooling Layers
from keras.layers import Flatten #To Flatten data for fully connected layers

#Convolutional Layer with One set of convolutional and pooling layers
#import dataset
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#We need to reshape the dataset to [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

#We need to normalize the pixel values to be from 0 to 1 (Done by dividing by the max value, 255)
X_train = X_train/ 255
X_test = X_test/ 255

#We now convert target variable to binary categories
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_test.shape #shape of y_test, 1000x10- 10 is the no. of classes
num_classes = y_test.shape[1]

#Defining a function that creates a model - we start with a set of convolutional and pooling layers
def convolutional_model():
    #Creating model
    model = Sequential()
    model.add(Conv2D(16, (5,5), strides=(1,1), activation='relu', input_shape=(28, 28, 1))) #16 filters with 5,5 parts and strides of 1,1
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) #MaxPooling2D layer with filter size 2,2 and strides 2,2 
    
    model.add(Flatten())
    model.add((Dense(100, activation='relu'))) #Adding fully connected layer with 100 neurons
    model.add(Dense(num_classes, activation='softmax')) #Adding input layer with size = to num_classes
    
    #Compile model
    model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
    return model

#lets call the function that builds the model and train it and evaluate its performance
model = convolutional_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=1)
scores = model.evaluate(X_test, y_test, verbose = 1)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))


'''

NOW WE WILL TRY A CONVNET with 2 two sets of Convolutional and Pooling layers

'''

def convolutional_model2():
    #Creating model
    model = Sequential()
    model.add(Conv2D(16, (5,5), strides=(1,1), activation='relu', input_shape=(28, 28, 1))) #16 filters with 5,5 parts and strides of 1,1
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) #MaxPooling2D layer with filter size 2,2 and strides 2,2 
    
    model.add(Conv2D(8, (2,2), activation='relu')) #2nd Convolutional layer with 8 filters
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add((Dense(100, activation='relu'))) #Adding fully connected layer with 100 neurons
    model.add(Dense(num_classes, activation='softmax')) #Adding input layer with size = to num_classes
    
    #Compile model
    model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
    return model


#lets call the function that builds the model and train it and evaluate its performance
model = convolutional_model2()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=1)
scores = model.evaluate(X_test, y_test, verbose = 1)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))


'''
Lets try a conventional neutral network

'''

(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images
X_train = X_train / 255
X_test = X_test / 255  
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


def conventional_model():
    
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
model = conventional_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=1)
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))    
    
    
    