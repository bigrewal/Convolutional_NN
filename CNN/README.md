# Convolutional Neural Networks Examples

### 1. CNN_mnist.py 
Uses CNN on MNIST data for classification of Handwritten digits.

#### Architecture:
	(Convolutional Layer F[5 * 5 * 1 * 32] --> RELU --> MAX_POOL (2*2) Stride = 2 )  *2 --> 
	
	FULLY_CONNECTED_LAYER[7*7*64,1024] --> DROP_OUT LAYER -->  FULLY_CONNECTED_LAYER[1024,10] --> Predicted Label
	
###### The above architecture is taken from https://www.tensorflow.org/get_started/mnist/pros 


#### It receives 97.6% accuracy on the Validation Set.


### 2. use_saved_Model.py 
Uses the trained CNN model to classify digits. The Model is saved in **savedModels/**  directory.