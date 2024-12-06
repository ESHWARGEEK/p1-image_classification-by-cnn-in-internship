from tensorflow import keras
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import load_model
from keras.preprocessing import image
import streamlit as st
from PIL import Image
import numpy as np

# Defining the CNN
cnn = keras.Sequential()
cnn.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same',activation='relu', input_shape=(32,32,3)))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())
cnn.add(Dense(64,activation='relu'))
cnn.add(Dropout(0.3))
cnn.add(Dense(10,activation='softmax'))

# Importing from CIFAR-10 dataset
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0  # Don't forget to normalize x_test as well

# Compile and train the model
cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Uncomment the next line to train the model
history = cnn.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save the model
cnn.save('cnn_model.h5')

