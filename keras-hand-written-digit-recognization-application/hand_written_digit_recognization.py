from keras import layers
from keras import models
# The MNIST dataset comes preloaded in Keras, in the form of a set of four Numpy arrays.
from keras.datasets import mnist
from keras.utils import to_categorical


# Step-1: loading the MNIST dataset in Keras
""""
train_images and train_labels form the training set, the data that the model will learn from. 
test_images and test_labels form the testing set, the data that the model will then be tested on.
"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

""""
The images are encoded as Numpy arrays, and the labels are an array of digits, ranging
from 0 to 9. The images and labels have a one-to-one correspondence.
"""
print('Training Set Shape: ', train_images.shape)
print('Training Set Length: ', len(train_labels))
print('Testing Set Shape: ', test_images.shape)
print('Testing Set Length: ', len(test_labels))

""""
The workflow will be as follows: First, we’ll feed the neural network the training data,
train_images and train_labels. The network will then learn to associate images and
labels. Finally, we’ll ask the network to produce predictions for test_images, and we’ll
verify whether these predictions match the labels from test_labels.
"""

# Step-2: The network architecture-construct the neural networks
""""
Here, our network consists of a sequence of two Dense layers, which are densely
connected (also called fully connected) neural layers. The second (and last) layer is a
10-way softmax layer, which means it will return an array of 10 probability scores (summing
to 1). Each score will be the probability that the current digit image belongs to
one of our 10 digit classes.
"""
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# Step-3: The compilation step-compile the neural networks model
""""
To make the network ready for training, we need to pick three more things, as part
of the compilation step:
 A loss function—How the network will be able to measure its performance on
the training data, and thus how it will be able to steer itself in the right direction.
 An optimizer—The mechanism through which the network will update itself
based on the data it sees and its loss function.
 Metrics to monitor during training and testing—Here, we’ll only care about accuracy
(the fraction of the images that were correctly classified).
"""
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Step-4: Preparing the image data
""""
Before training, we’ll preprocess the data by reshaping it into the shape the network
expects and scaling it so that all values are in the [0, 1] interval. Previously, our training
images, for instance, were stored in an array of shape (60000, 28, 28) of type
uint8 with values in the [0, 255] interval. We transform it into a float32 array of
shape (60000, 28 * 28) with values between 0 and 1.
"""
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Step-5: Preparing the labels-encode the labels categorically
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Step-6: train the network
""""
We’re now ready to train the network, which in Keras is done via a call to the network’s
fit method—we fit the model to its training data.
"""
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Step-7: finally, check the accuracy of the network over the training data
""""
Two quantities are displayed during training: the loss of the network over the training
data, and the accuracy of the network over the training data.
We quickly reach an accuracy of 0.989 (98.9%) on the training data.
"""
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test Accuracy: ', test_acc)
