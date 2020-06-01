"""
An Unsupervised deep learning model, the autoencoder is a data compression algorithm where
the compression and the decompression functions are learned automatically from data. 

Instead of being engineered by a human. Such autoencoders are built using neural networks. 
Autoencoders are data specific, which means that they will only be able to compress data similar to what they have been trained on. 
An autoencoder trained on pictures of cars would do arather poor job of compressing pictures of buildings.


Some interesting applications of autoencoders are data denoising and dimensionality reduction for data visualization. 

An Autoencoder takes an image, for example, as an input and uses an encoder to find the optimal compressed
representation of the input image. Then, using a decoder the original image is restored. 

So an autoencoder is an unsupervised neural network model. It uses backpropagation by setting the target variable to be the same as the input. 

Because of non-linear activation functions in neural networks, 
autoencoders can learn data projections that are more interesting than a principal component analysis PCA or other basic techniques,
which can handle only linear transformations. 

A very popular type of autoencoders is the Restricted Boltzmann Machines or (RBMs) for short. 
RBMs have been successfully used for various applications, including fixing imbalanced datasets. 
Because RBMs learn the input in order to be able to regenerate it, then they can learn the distribution of the minority class in an imbalance dataset,
and then generate more data points of that class, transforming the imbalance dataset into a balanced data set.

Similarly, RBMs can also be used to estimate missing values in different features of a data set. 
Another popular application of Restricted Boltzmann Machines is automatic feature extraction of especially unstructured data. 


"""

