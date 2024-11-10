from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
import numpy as np

# Load pre-trained ResNet50 model + weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Function to extract features from an image
def extract_image_features(img_array):
    # Expand the dimensions to match the model's expected input
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess image for ResNet50

    # Extract features using the pre-trained ResNet50 model
    features = base_model.predict(img_array)

    # Flatten the features to make them compatible with the dense layers
    features_flattened = features.flatten()

    return features_flattened


# Example: Extract features from a property image
img_path = 'path_to_property_image.jpg'
img_array = preprocess_image(img_path)  # Process the image
image_features = extract_image_features(img_array)  # Extract features
