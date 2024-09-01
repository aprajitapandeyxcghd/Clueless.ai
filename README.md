# Fashion-Recommendation-System
---

# Fashion Recommendation System

This project is a **Fashion Recommendation System** built using TensorFlow and Streamlit. It allows users to upload an image of a fashion item, and the system will recommend similar items based on visual features extracted from a pre-trained ResNet50 model.

## Features

- **Image Feature Extraction**: Utilizes the ResNet50 model, pre-trained on ImageNet, to extract meaningful features from fashion images.
- **Similarity Matching**: Implements a k-Nearest Neighbors (k-NN) algorithm to find the closest matches to the uploaded image based on the extracted features.
- **Interactive User Interface**: Provides a user-friendly interface using Streamlit, where users can upload images and view recommended items in real-time.

## How It Works

1. **Feature Extraction**: 
   - Images are preprocessed and passed through the ResNet50 model to extract deep features.
   - The features are then normalized for consistent distance calculations.
   
2. **Finding Nearest Neighbors**: 
   - A k-NN model is trained on the extracted features of a dataset of fashion images.
   - When a user uploads an image, its features are compared against the dataset, and the most similar images are recommended.

3. **User Interaction**:
   - Users can upload an image via the Streamlit interface.
   - The system displays the uploaded image and the top 5 most similar images from the dataset.

## Files

- `app.py`: The main application file that runs the Streamlit interface and handles image upload and recommendation logic.
- `Images_features.pkl`: Precomputed features of the images in the dataset.


## Dependencies

- TensorFlow
- NumPy
- Scikit-learn
- Streamlit
- Pillow

