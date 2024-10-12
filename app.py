import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st

# Set the page configuration
st.set_page_config(page_title="Fashion Image Recommendation", layout="wide")

# Title of the web app
st.title('Fashion Image Recommendation System')

# Load pre-saved image features and filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Function to extract features from the uploaded image
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Set up the nearest neighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Image uploader in the Streamlit UI
upload_file = st.file_uploader("Upload an image to find similar fashion items", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    # Save the uploaded file to a temporary location
    file_path = os.path.join('upload', upload_file.name)
    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())
    
    # Display the uploaded image
    st.subheader('Uploaded Image')
    st.image(upload_file)

    # Extract features from the uploaded image
    input_img_features = extract_features_from_images(file_path, model)

    # Find similar images based on the uploaded image
    distance, indices = neighbors.kneighbors([input_img_features])

    # Display the recommended images
    st.subheader('Recommended Images')
    cols = st.columns(5)  # Create 5 columns to display 5 images in a row
    for i, col in enumerate(cols):
        col.image(filenames[indices[0][i+1]])  # Display the recommended images in columns
