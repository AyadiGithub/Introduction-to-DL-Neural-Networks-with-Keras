import numpy as np

#Creating a neural network with 1 hidden layer and 6 weights and 3 biases
weights = np.around(np.random.uniform(size=6), decimals=2)
biases = np.around(np.random.uniform(size=3), decimals=2)
print(weights)
print(biases)

#lets create x1,x2 for the input layer
x_1 = 0.5
x_2 = 0.85
print('x1 is {} and x2 is {}'.format(x_1,x_2))

#setting up formula for hidden layer
z_11 = x_1*weights[0]+x_2*weights[1]+biases[0]
z_12 = x_1*weights[2]+x_2*weights[3]+biases[1]

print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_11, decimals=4)))
print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=4)))

#Let's use a sigmoid activation function
#We import math and define sigmoid function
import math 
def sigmoid(x):
  return  1.0 / (1.0 + np.exp(-1 * x))

#Lets compute the activation of the first and second nodes in the hidden layer
a_11 = sigmoid(z_11)
a_12 = sigmoid(z_12)
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))

#Lets compute the weighted sum for the output layer node
z_2 = a_11*weights[4]+a_12*weights[5]+biases[2]
print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))

#Lets use the sigmoid function to compute the output of the network a_2
a_2 = sigmoid(z_2)
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))


'''

In order to code an automatic way of making predictions, let's generalize our network. 
A general network would take 𝑛 inputs, would have many hidden layers, each hidden layer having 𝑚 nodes, and would have an output layer. 
We will code the network to have many hidden layers. 
We will code the network to have more than one node in the output layer.

'''

#Initializing a Network

n = 2 #Number of Inputs for the layer
num_hidden_layers = 2 #Number of hidden layers
m = [2,2] #Number of nodes in each hidden layer
num_nodes_output = 1 #Number of nodes in the output layer

#Now we initialize the weights and biases to random numbers using Numpy
num_nodes_previous = n #Number of nodes in the previous layer
network = {} #Initializing a network as an empty dictionary


#Lets make a loop that goes through each layer and randomly initializes weights and biases for each node
#We need to add +1 to number of hidden layers to account for the output layer

for layer in range (num_hidden_layers +1):
    
    #determining the name of layer
    if layer == num_hidden_layers:
        #print(layer) #Visualizing what is happening in the loop
        layer_name = 'output'
        num_nodes = num_nodes_output
    else:
        #print(layer) #Visualizing what is happening in the loop
        layer_name = 'layer_{}'.format(layer + 1)
        num_nodes = m[layer]
    
    #initializing weights and outputs for each node in current layer
    network[layer_name]= {}
    for node in range (num_nodes):
        #print(num_nodes) #Visualizing what is happening in the loop
        node_name = 'node_{}'.format(node+1)
        network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
            }

    num_nodes_previous = num_nodes

print(network) #showing the network


#Lets put the loop that we made in a function. This way we can create different NN with one line of code. 
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    
    num_nodes_previous = num_inputs # number of nodes in the previous layer

    network = {}
    
    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range (num_hidden_layers +1):
    
        #determining the name of layer
        if layer == num_hidden_layers:
        #print(layer) #Visualizing what is happening in the loop
            layer_name = 'output'
            num_nodes = num_nodes_output
        else:
        #print(layer) #Visualizing what is happening in the loop
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = num_nodes_hidden[layer]
    
        #initializing weights and outputs for each node in current layer
        network[layer_name]= {}
        for node in range (num_nodes):
        #print(num_nodes) #Visualizing what is happening in the loop
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes

    return network # return the network

#Creating a small NN using the function 'initialize_network' that we created
small_network = initialize_network(5, 3, [3, 2, 3], 1)
small_network
#Small_network has 5 inputs and 3 hidden layers
#3 nodes in the 1st layer, 2 nodes in the 2nd layer, and 3 nodes in the 3rd layer
#1 node in the output layer


#Now we need to compute the weighted sum at each Node
#We create a function to perform this.

def compute_weighted_sum(inputs,weights,bias):
    return np.sum(inputs * weights) + bias


#Lets generate 5 inputs to feed into our NN small_network
from random import seed
np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)
print('The inputs to the network are {}'.format(inputs))

#Lets compute the weighted sum for the 1st node in the first hidden layer
#We navigate to weights in node 1 in layer 1 in small network
node_weights = small_network['layer_1']['node_1']['weights']
#We navigate to bias in node 1 in layer 1 in small network
node_bias = small_network['layer_1']['node_1']['bias']
#We assign the weights and bias we extracted to the compute_weighted_sum function
weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)))

#We define a Computer Node Activate function to be a sigmoid function
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

#Computing Output of the first node
node_output  = node_activation(weighted_sum)
print('The output of the first node in the first hidden layer is {}'.format(np.around(node_output[0], decimals=4)))


'''
# OKAY SO NOW LETS COMBINE COMPUTED_WEIGHTED_SUM AND NODE_ACTIVATION functions


1. We Start with the input layer as the input to the first hidden layer.
2. We Compute the weighted sum at the nodes of the current layer.
3. We Compute the output of the nodes of the current layer.
4. We Set the output of the current layer to be the input to the next layer.
5. We Move to the next layer in the network.
6. We Repeat steps 2 - 4 until we compute the output of the output layer.


'''


def forward_propagate(network, inputs):
    
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    #print(layer_inputs)
    for layer in network:
        #print(layer)
        layer_data = network[layer]
        
        
        layer_outputs = [] 
        for layer_node in layer_data:
           #print(layer_node)
            node_data = layer_data[layer_node]
        
            # compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
            
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
    
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer
       #print(layer_outputs)
    network_predictions = layer_outputs
    return network_predictions

predictions = forward_propagate(small_network, inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))


#Lets create a new network called my_deep_network
my_deep_network = initialize_network(5,10,[12,12,12,12,12,10,10,10,8,8],3)
inputs = np.around(np.random.uniform(size=5), decimals=2)
predictions = forward_propagate(my_deep_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))
