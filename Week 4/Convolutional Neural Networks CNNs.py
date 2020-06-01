'''
CONVOLUTIONAL NEURAL NETWORKS - CNNs

CNNs make the excplicit assumption that the inputs are images.
This allows us to incorporate certain properties into their architecture.
These properties make forward propagation more efficient and reduces no. of parameters in the Network. 
Applications for CNNs are: Image recognition, object detection and other Computer Vision applications.

Example CNN:
    Input Image --> 1.Convolution Layer --> 2.Pooling Layer --> 3.Convolution Layer --> 4.Pooling Layer --> 5.Fully Connected Layer --> Output

In Convolution networks, we use RGB filters on images (n x m x 3(RGB)) to preserve the spatial dimensions and to drastically reduce the number of parameters. 
Decreasing the number of parameters also helps prevent the the model from overfitting the training data. 

A Convolutional layer also consists of ReLU which filter the output (-ve numbers become 0)
The next layer in our convolutional neural network is the pooling layer. 
The pooling layer's main objective is to reduce the spatial dimensions of the data propagating through the network.
2 Types of pooling: Max and average
    Max: each area scanned, we keep the highest values
    Average: we average the values of the scanned area

MaxPooling provides spatial variance which enables the neural network to recognize objects in an image even if
the object does not exactly resemble the original object.

Finally, in the fully connected layer, we flatten the output of the last convolutional layer and 
connect every node of the current layer with every other node of the next layer.
This Layer outputs an n-dimensional vector, where n is the number of classes pertaining to the problem at hand.

For example, if you are building a network to classify images of digits, the dimension n would be 10, since there are 10 digits. 




'''


#Building a Convolutional Neural Network in Keras
#Sequential constructor 

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten

model = Sequential()
input_shape = (128, 128, 3) #Define input shape for images as 128x128 pixels x 3 (3 = RGB)

#We add layers to the network
model.add(Conv2D(16, kernel_size=(2,2), strides=(1,1), activation='relu'), input_shape=input_shape)
#1st layer is a Convolutional layer with 16 filters, each filter is size 2x2 and slice through image in strides 1,1 (1 stride vertical, 1 stride horizontal)
#The layer uses the ReLU activaiton function

#Now we need a pooling layer - we use Max-Pooling - filter/pooling size of 2 and strides 2
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#3rd layer is a Convolutional layer with 32 filters, sized (2,2) and strides (2,2)
model.add(Conv2D(32, kernel_size=(2,2), strides=(2,2)))

#4th Layer - Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2))

#Now we flatten the layer
model.add(Flatten())

#Next is the fully connected Layer with 100 nodes
model.add(Dense(100, activation='relu'))

#Next is output layer with activation function 'Softmax' to convert outputs to probabilities. 
model.add(Dense(num_classes, activation='softmax'))          


