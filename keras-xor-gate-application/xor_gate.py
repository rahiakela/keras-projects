import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


# initialize a random seed, so that you can reproduce the same results when running the program again
np.random.seed(444)

""""
 define an X array, containing the 4 possible A-B sets of inputs for the XOR operation 
 and a y array, containing the outputs for each of the sets of inputs defined in X.
"""
# input: XOR operation
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# output: A XOR B
y = np.array([[0], [1], [1], [0]])

# define the neural network
model = Sequential()  # provided by keras to define a neural network, in which the layers of the network are defined in a sequential way
model.add(Dense(2, input_dim=2))  # first layer of neurons, composed of two neurons, fed by two inputs
model.add(Activation('sigmoid'))  # defining their activation function as a sigmoid function in the sequence
model.add(Dense(1))  # second or output layer, composed of neurons 1
model.add(Activation('sigmoid'))

#  training of the network
""""
To adjust the weights of the network, you’ll use the Stochastic Gradient Descent (SGD) 
with the learning rate equal to 0.1, and you’ll use the mean squared error as a loss function to be minimized.
"""
sgd = SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd)

""""
Finally, we perform the training by running the fit() method, using X and y as training examples 
and updating the weights after every training example is fed into the network (batch_size=1). 
The number of epochs represents the number of times the whole training set will be used to train the neural network.
Here, we are repeating the training 5000 times using a training set containing 4 input-output examples. 
By default, each time the training set is used, the training examples are shuffled.
"""
model.fit(X, y, batch_size=1, epochs=5000)

if __name__ == "__main__":
    # after the training process has finished, we print the predicted values for the 4 possible input examples
    print(model.predict(X))

""""
Output:
[[0.05873257]
 [0.94682413]
 [0.93230104]
 [0.0515943 ]]
As we defined X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), the expected output values are 0, 1, 1, and 0, 
which is consistent with the predicted outputs of the network, given we should round them to obtain binary values.
"""

