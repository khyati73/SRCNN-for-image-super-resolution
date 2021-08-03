# SRCNN-for-image-super-resolution
 High resolution images offer a high pixel density image thereby giving more details about the original image. High resolution images from original images is a necessity in the field of computer vision for better performance in pattern recognition and data analysis.
 Linear network : These have a very simple path or say a single path that does not consist of skip connections or multiple branches. Architecture of any CNN model:  CNN consists of an input layer, hidden layers and an output layer. In hidden layers inputs and outputs are masked by the activation function and final convolution activation function of hidden layers is commonly Relu. The activation function of a node defines the output of that node given an input or set of inputs. Then comes the pooling layers, fully connected layers and normalisation layers. 

Pooling: Function is to progressively reduce the spatial size of Representation to reduce the amount of parameters and computation in the network. It reduces the dimensions of the data by combining the outputs of neuron clusters at one layer into a single neuron in the next layer. Max pooling uses the maximum value of each cluster of neurons at the prior layer. Average pooling instead uses the average value. Pooling layers are added after the convolutional layer to reduce the dimensions of the feature maps.
Activation Function: Controls the amplitude. It defines the output of node and introduces nonlinear properties to the network. It calculates the weighted sum of input, adds a bias and then decides whether it should be fired or not.
In linear networks, several convolutional layers are stacked on top of each other and input flows sequentially from initial to later layers. Some linear networks learn to reproduce the residual image that is the difference between the LR and HR images. 

Super resolution convolutional neural network consists of convolutional layers where except the one each layer is followed by rectified linear unit non-linearity. There are a total of 3 convolutional and 2 Relu layers Stacked together linearly. The layers are named according to the functions performed by them. The first convolutional layer is termed as patch extraction or feature extraction which creates feature maps from input images. The second convolutional layer is called non linear mapping which converts the feature maps onto high dimensional feature vectors. The last convolutional layer aggregates the feature maps to output the final high resolution image. 
Bipolar cubic interpolated image is an Input and the network produces the same size resolution image as output. Convolutional layer I:  in this layer patch extraction will be performed. Patch extraction is a process of selecting the patch that is a set of pixels in the image. Convolutional layer II:  In this layer non linear mapping is performed and relu is used as activation function here,  which returns 0 if it receives a negative input.  Pooling is performed in the second layer. Convolutional layer III:  Reconstruction of image takes place here.
The training data set is synthesized by extracting non-overlapping dense patches of size 32 X 32 from the HR images. The LR input patches are first downsampled and then up-sampled using bicubic interpolation having the same size as the high resolution output image.

Weights: Each neuron  is a neural network that computes an output value by applying a specific function to the input values coming from the receptive field in the previous layer. The function that is applied to the input values is determined by a vector of weights and the bias. The vector of weights and bias are called filters and represent particular features of the input.

Activation function: When discussed about activation function; a number of activation functions exist. Some activation functions are sigmoid function, hyperbolic tangent, ReLu function, leaky ReLu, Exponential linear unit and parametric ReLu. Out of which, we intend to use the ReLu activation function. We prefer Relu over other activation functions because it does not activate all the neurons at the same time. And the purpose of using activation function is because it should help to decide if the neuron will be fired or not.
About The dataset
I have taken up a zip file of a dataset called yang91.
Can be downloaded from here.
https://drive.google.com/file/d/1AEjhNY6LOdACZAXdowfJ-phINx4NaGbP/view?usp=sharing
