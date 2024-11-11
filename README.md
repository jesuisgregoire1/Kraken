# Property Evaluation and Description Automation

## Overview

This project automates the evaluation and description of properties using image data (interior photos) and location data (address/coordinates). The goal is to build a machine learning model that takes images of properties and their associated location information to output a detailed property description. This includes predicting the room count, room types (e.g., kitchen, bedroom), and potential property value.

## Features

- **Image Processing**: Uses pre-trained models (e.g., ResNet50) for extracting features from property images, which helps in classifying room types and identifying property characteristics.
- **Location Data**: Incorporates geographic information (latitude, longitude) to better understand the property’s market value, based on location.
- **Room and Property Type Prediction**: The model predicts the room count and types (e.g., number of bedrooms, kitchens, bathrooms) based on images and location.
- **Price Estimation**: The model estimates potential property value using the images and location features.
- **Data Structure**: The project uses a **Pandas DataFrame** to store and manage property data, including text and image features.

## Requirements

- Python 3.8 or higher
- TensorFlow or PyTorch (for image processing)
- OpenCV (for image feature extraction)
- Geopy (for geolocation data)
- Pandas (for managing property data)
- Matplotlib (optional, for visualizations)
- NumPy (for array manipulation)

## Installation

1. Clone the repository:

   ```bash
      git clone https://github.com/jesuisgregoire1/Kraken.git
      cd Kraken
      pip install -r requirements.txt  # Install dependencies

      # To generate a new model and plot graphs
      python3 main.py

      # To run the web application (assuming it's a Flask/Django app)
      python3 app.py

## Data Structure

1. The dataset consists of the following columns:

- **address**: Property address
- **price**: Property price 
- **rooms**: Total number of rooms 
- **area**: Property area 
- **bathrooms**: Number of bathrooms 
- **kitchens**: Number of kitchens 
- **images**: List of image file paths
- **image_features**: Processed image features 
- **lat**: Latitude 
- **lon**: Longitude 
- **lat_normalized**: Normalized latitude 
- **lon_normalized**: Normalized longitude 

## Model Overview

The project uses ResNet50 (pre-trained on ImageNet) for feature extraction from property images. These features are averaged across multiple images of the same property for a more accurate description. Geopy is used to process location data and normalize coordinates for more consistent predictions.

Model Workflow
Image Preprocessing: The images are preprocessed (resized, normalized, etc.) before being input into the ResNet50 model for feature extraction.
Feature Aggregation: Features from multiple images are averaged to create a single feature vector for each property.
Room Count Prediction: The model predicts the total number of rooms based on image features and location.
Room Type Classification: Based on extracted features, the model classifies room types (e.g., bedroom, kitchen, bathroom).
Price Estimation: The model estimates the potential property value based on location and image features.