'''

Recurrent neural networks or (RNNs) for short, are networks with loops that don't just take a new input at a time,
but also take in as input the output from the previous data point that was fed
into the network. 

Essentially in a Recurrent Neural Network, we can start with a normal neural network.
At time t = 0, the network takes in input x0 and outputs a0. 
Then, at time t = 1, in addition to the input x1, the network also takes a0 as input,
weighted with weight w0,1, and so on and so forth. 

Recurrent Neural Networks are good at modelling patterns and sequences of data, such as:
texts, genomes, handwriting, and stock markets. 

These algorithms take time and sequence into account, which means that they have a temporal dimension. 

Long short-term memory model or LSTM model for short is a very popular type of RNNs. 

It has been successfully used for many applications including:
Image generation, where a model trained on many images is used to generate new novel images. 

Another application is handwriting generation.

LSTM models have also been successfully used to build algorithms that can automatically describe (caption) images as well as streams of videos. 


'''