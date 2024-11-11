import numpy as np
import tensorflow as tf
import os
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras._tf_keras.keras.applications import ResNet50


class ImageProcessing:
    def __init__(self):
        # initialize ResNet50 model for extraction of features
        self._resnet_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

        # Initialize ImageDataGenerator for data augmentation
        self.datagen = ImageDataGenerator(
            rotation_range=40,  # randomly rotate images
            width_shift_range=0.2,  # shift images horizontally
            height_shift_range=0.2,  # shift images vertically
            shear_range=0.2,  # shear
            zoom_range=0.2,  # zoom into
            horizontal_flip=True,  # randomly flip the images horizontally
            fill_mode='nearest'  # fill the pixels after transformation
        )

    def get_image_features(self, img_path):
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            return None

        img = load_img(img_path, target_size=(224, 224))  # resize the image (224, 224)
        img_array = img_to_array(img)  # convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # preprocess input for ResNet50

        augmented_images = self.datagen.flow(img_array, batch_size=1)  # generate augmented images

        features = self._resnet_model.predict(augmented_images[0])  # predict features
        return features.flatten()

    def process_property_images(self, properties_df):
        properties_df['image_features'] = None  # initialize a column for image features
        for idx, row in properties_df.iterrows():
            images = row['images']
            features_for_property = []
            for img_path in images:
                features = self.get_image_features(img_path)  # extract features for each image
                if features is not None:
                    features_for_property.append(features)

            # average features for all images of this property
            if features_for_property:
                avg_features = np.mean(features_for_property, axis=0)
                properties_df.at[idx, 'image_features'] = avg_features  # store average features in DataFrame
        return properties_df
