# Import required packages
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Step-1: Getting the data ready
""""
Generate train dummy data for 1000 Students and dummy test for 500
Columns :Age, Hours of Study & Avg Previous test scores
"""
np.random.seed(2018)  # Setting seed for reproducibility
train_data = np.random.random((1000, 3))
test_data = np.random.random((500, 3))

# Step-2: Prepare the data
# Generate dummy training dataset for 1000 students : Whether Passed (1) or Failed (0)
labels = np.random.randint(2, size=(1000, 1))

# Step-3: Construct the model structure
""""
Defining the model structure with the required layers, # of
neurons, activation function and optimizers
"""
model = Sequential()
model.add(Dense(5, input_dim=3, activation='relu'))  # adding first hidden layer with input data, activation function and specifying neurons
model.add(Dense(4, activation='relu'))  # adding second hidden layer with activation function and specifying neurons
model.add(Dense(1, activation='sigmoid'))  # adding output layer with activation function and specifying neurons

# Step-4: Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step-5: Training the Model and Making Predictions
""""
Once the network is defined, we can use the training data with the correct
predictions to train the network using the “fit” method for the model.
"""
model.fit(train_data, labels, epochs=10, batch_size=32)

# Step-6: Check the accuracy, by making predictions from the trained model
""""
Finally, once the model is trained, we can use the trained model to make
predictions on the new test dataset.
"""
predictions = model.predict(test_data)
print(predictions)
