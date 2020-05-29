'''
Activation functions play a major role in the learning process of a NN. 
Sigmoid function has its shortcomings since it can lead to the vanishing gradient problem for the earlier layers. 
Lets talk about some activation functions that are more efficient and more applicable for DL Applications

There are 7 types of Activation Functions that can be used when building a NN. 
1. Binary step function 
2. Linear/Identity function
3. Sigmoid/Logistic funciton
4. Hyperbolic Tangent function (TanH)
5. Rectified Linear Unit function (ReLU)
6. Leaky ReLU
7. Softmax function

For these 7 funcitons, the Popular ones are:
    - Sigmoid
    - Hyperbolic Tangent
    - ReLU
    - Softmax





#Sigmoid 

a = 1/(1 + exp(-z))
Problems: Vanishing gradient as function approaches z = -3 or +3 regions 
          Function limited to 1 and 0
          Not symmetric around origin
          

#Hyperbolic Tangent (TanH)
a = (exp(z) - exp(-z))/(exp(z) + exp(-z))
Similar to Sigmoid. A scaled version of it or stretched but symmetric over the origin. 
Ranges from -1 to +1

Problems: Vanishing gradient problem in very deep NN. 


#Rectified Linear Unit (ReLU)
Most widely used Activation function today. Only in Hidden Layers
a = max(0,z)
Non-linear function
Advantage : Doesn't activate all neurons at the same time.
            If input is -ve, it is converted to 0 and neuron not activated. 
            At a time, only a few neurons are activated, so network is sparse and efficient. 
            No vanishing gradient problem


#Softmax Function
A type of sigmoid functions and handy for classification problems. 
a(i) = exp(z(i))/sum(exp(z(k)))
np.exp(x) / np.sum(np.exp(x), axis=0)


Ideally used in the output layer of the classifier in order to get probabilites to define the class of each input. 


'''


