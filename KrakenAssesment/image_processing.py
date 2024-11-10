import keras._tf_keras.keras.applications
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras._tf_keras.keras.applications import ResNet50
import os

# Define ResNet50 model for feature extraction
resnet_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')


def get_image_features(img_path):
    """Load, preprocess, and extract features from a single image using ResNet50."""
    if not os.path.exists(img_path):
        print(f"Warning: Image not found at {img_path}")
        return None

    # Load and resize the image (224, 224), as expected by ResNet50
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)  # Convert image to numpy array

    # Preprocess the image for ResNet50
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Preprocess input for ResNet50

    # Extract features using ResNet50
    features = resnet_model.predict(img_array)

    return features.flatten()


def process_property_images(properties_df):
    """Process all images for each property and extract features."""
    properties_df['image_features'] = None  # Initialize a column for image features

    for idx, row in properties_df.iterrows():
        images = row['images']
        features_for_property = []

        for img_path in images:
            # Extract features for each image
            features = get_image_features(img_path)
            if features is not None:
                features_for_property.append(features)

        # Compute average features for all images of this property
        if features_for_property:
            avg_features = np.mean(features_for_property, axis=0)
            properties_df.at[idx, 'image_features'] = avg_features  # Store average features in DataFrame

    return properties_df