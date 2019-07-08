#Let’s start off by importing the classes and functions we will need.
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import cv2
import matplotlib.pyplot as plt


#It is always a good idea to initialize the random number generator to a constant to ensure that the results of your script are reproducible.
seed = 7
numpy.random.seed(seed)

#Now we can load the MNIST dataset using the Keras helper function
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

""" The training dataset is structured as a 3-dimensional array of instance, image width and image height. For a multi-layer perceptron model we must reduce the 
images down into a vector of pixels. In this case the 28×28 sized images will be 784 pixel input values. We can do this transform easily using the reshape() function 
on the NumPy array. We can also reduce our memory requirements by forcing the precision of the pixel values to be 32 bit, the default precision used by Keras anyway. """
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

""" The pixel values are gray scale between 0 and 255. It is almost always a good idea to perform some scaling of input values when using neural network models.
 Because the scale is well known and well behaved, we can very quickly normalize the pixel values to the range 0 and 1 by dividing each value by the maximum of 255. """
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

""" Finally, the output variable is an integer from 0 to 9. This is a multi-class classification problem. As such, it is good practice to use a one hot encoding 
of the class values, transforming the vector of class integers into a binary matrix.
We can easily do this using the built-in np_utils.to_categorical() helper function in Keras. """
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


""" We are now ready to create our simple neural network model.
 We will define our model in a function. This is handy if you want to extend the example later and try and get a better score. """
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

""" The model is a simple neural network with one hidden layer with the same number of neurons as there are inputs (784). A rectifier activation function is used for the neurons in the hidden layer.
A softmax activation function is used on the output layer to turn the outputs into probability-like values and allow one class of the 10 to be selected as the model’s output prediction. Logarithmic loss is used as the loss function (called categorical_crossentropy in Keras) and the efficient ADAM gradient descent algorithm is used to learn the weights.
We can now fit and evaluate the model. The model is fit over 10 epochs with updates every 200 images. The test data is used as the validation dataset, allowing you to see the skill of the model as it trains. A verbose value of 2 is used to reduce the output to one line for each training epoch.
Finally, the test dataset is used to evaluate the model and a classification error rate is printed. """

"""
# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=100, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model.save('net2.h5')
"""



