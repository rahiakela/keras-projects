""""
For our experiment, let’s select a popular Keras dataset for developing
a model. We can start with the Boston House Prices dataset. It is taken
from the StatLib library, which is maintained at Carnegie Mellon
University. The data is present in an Amazon S2 bucket, which we can
download by using simple Keras commands provided exclusively for the datasets.
"""
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers  import Dense, Activation
import numpy as np


# Download the data using Keras; this will need an active internet connection
(x_train, y_train),(x_test, y_test) = boston_housing.load_data()

# Explore the data structure using basic python commands
print('Type of the Dataset:', type(y_train))
print('Shape of training data :', x_train.shape)
print('Shape of training labels :', y_train.shape)
print('Shape of testing data :', type(x_test))
print('Shape of testing labels :', y_test.shape)

print(x_train[:3, :])

# Extract the last 100 rows from the training data to create the validation datasets.
x_val = x_train[300:,]
y_val = y_train[300:,]

# Define the model architecture
model = Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_val, y_val))

# Evaluate model and to study the results of the model
results = model.evaluate(x_test, y_test)
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i], ' : ', results[i])

