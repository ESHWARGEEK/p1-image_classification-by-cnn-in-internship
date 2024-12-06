import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import load_model
from keras.preprocessing import image
import streamlit as st
from PIL import Image
import numpy as np

def cifar10_classification():
    st.title("cifar-10 classification of image using cnn")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.mode != 'RGB':
            image = image.conver('RGB')

        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write("Classifying...")

        model = load_model('cnn_model.h5')

        class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


        img = image.resize((32,32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_classs = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        st.write(f'prediction : {class_name[predicted_classs]}')
        st.write(f'confidence: {"{:.2f}".format(confidence*100)}%')

cifar10_classification()