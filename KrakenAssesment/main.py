#step 1
# extract images and text from the PDF
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.src.layers import Dropout
from data_extraction import DataExtraction
import location_processing
import image_processing
import os
import numpy as np
from keras.api.models import Model, load_model
from keras.api.layers import Input, Dense, Concatenate, Flatten
from keras.api.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

path_to_pdf = os.path.abspath("Files/Real Estate Properties.pdf")

pdf_path = 'path/to/your/pdf_file.pdf'

# Create an instance of the DataExtraction class
data_extractor = DataExtraction(path_to_pdf)

# Extract the property data into a DataFrame
properties_df = data_extractor.extract_to_dataframe()

# Optionally, you can save the DataFrame to a CSV file
#properties_df.to_csv('extracted_property_data.csv', index=False)

# Display the DataFrame (optional)
print(properties_df.head())

properties_df = image_processing.process_property_images(properties_df)
c = 1
properties_df = location_processing.process_location_data(properties_df)
c = 1
#properties_df['Processed_Images'] = processed_images
#print(properties_df.head())

X_image = np.vstack(properties_df['image_features'].values).astype('float32')  # Stack and ensure float32 type
X_location = properties_df[['lat', 'lon']].values.astype('float32')  # Ensure float32 type for location data

# Step 4: Extract and preprocess target variables
y_room_count = properties_df['rooms'].values.astype('float32')  # Room count as float32
y_kitchens = properties_df['kitchens'].values.astype('float32')  # Kitchens as float32
y_bathrooms = properties_df['bathrooms'].values.astype('float32')  # Bathrooms as float32
y_value = properties_df['price'].apply(lambda x: float(x.replace('€', '').replace(',', '').strip())).values.astype('float32')

# Normalize the price (optional)
y_value = (y_value - np.mean(y_value)) / np.std(y_value)

# Step 5: Define model
# Image input
image_input = Input(shape=(2048,))
x1 = Dense(128, activation='relu')(image_input)
x1 = Dropout(0.3)(x1)  # Dropout layer to prevent overfitting
x1 = Dense(64, activation='relu')(x1)
x1 = Dropout(0.3)(x1)  # Another Dropout layer

# Location input with Dropout
location_input = Input(shape=(2,))
x2 = Dense(64, activation='relu')(location_input)
x2 = Dropout(0.3)(x2)
x2 = Dense(32, activation='relu')(x2)
x2 = Dropout(0.3)(x2)

# Concatenate branches
merged = Concatenate()([x1, x2])

# Define outputs
room_count_output = Dense(1, activation='linear', name='room_count')(merged)
kitchens_output = Dense(1, activation='linear', name='kitchens')(merged)
bathrooms_output = Dense(1, activation='linear', name='bathrooms')(merged)
value_output = Dense(1, activation='linear', name='value')(merged)

# Model definition
model = Model(inputs=[image_input, location_input], outputs=[room_count_output, kitchens_output, bathrooms_output, value_output])

# Compile model
model.compile(optimizer=Adam(),
              loss={'room_count': 'mean_squared_error',
                    'kitchens': 'mean_squared_error',
                    'bathrooms': 'mean_squared_error',
                    'value': 'mean_squared_error'},
              metrics={'room_count': 'mae',
                       'kitchens': 'mae',
                       'bathrooms': 'mae',
                       'value': 'mae'})

# Model summary
model.summary()

# Step 6: Train/test split
X_image_train, X_image_test, X_location_train, X_location_test, y_room_count_train, y_room_count_test, y_kitchens_train, y_kitchens_test, y_bathrooms_train, y_bathrooms_test, y_value_train, y_value_test = train_test_split(
    X_image, X_location, y_room_count, y_kitchens, y_bathrooms, y_value, test_size=0.2, random_state=42)

# Step 7: Train the model

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
]

history = model.fit(
    [X_image_train, X_location_train],
    {'room_count': y_room_count_train,
     'kitchens': y_kitchens_train,
     'bathrooms': y_bathrooms_train,
     'value': y_value_train},
    epochs=50,
    batch_size=32,
    validation_data=([X_image_test, X_location_test],
                     {'room_count': y_room_count_test,
                      'kitchens': y_kitchens_test,
                      'bathrooms': y_bathrooms_test,
                      'value': y_value_test}),
    callbacks=callbacks,
    verbose=1
)

model.save('best_model.keras')

model = load_model('best_model.keras')

# Step 9: Evaluate the model on the test set
test_loss, test_room_count_loss, test_kitchens_loss, test_bathrooms_loss, test_value_loss, test_room_count_mae, test_kitchens_mae, test_bathrooms_mae, test_value_mae = model.evaluate(
    [X_image_test, X_location_test],
    {'room_count': y_room_count_test,
     'kitchens': y_kitchens_test,
     'bathrooms': y_bathrooms_test,
     'value': y_value_test},
    verbose=1
)

# Print the evaluation results
print("Test Loss:", test_loss)
print("Room Count MAE:", test_room_count_mae)
print("Kitchens MAE:", test_kitchens_mae)
print("Bathrooms MAE:", test_bathrooms_mae)
print("Price MAE:", test_value_mae)

# Step 10: Plot the training history (loss and metrics)
# Training history
plt.figure(figsize=(12, 6))

# Loss plots
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# MAE plots
plt.subplot(1, 2, 2)
plt.plot(history.history['room_count_mae'], label='Room Count MAE')
plt.plot(history.history['val_room_count_mae'], label='Validation Room Count MAE')
plt.plot(history.history['kitchens_mae'], label='Kitchens MAE')
plt.plot(history.history['val_kitchens_mae'], label='Validation Kitchens MAE')
plt.plot(history.history['bathrooms_mae'], label='Bathrooms MAE')
plt.plot(history.history['val_bathrooms_mae'], label='Validation Bathrooms MAE')
plt.plot(history.history['value_mae'], label='Price MAE')
plt.plot(history.history['val_value_mae'], label='Validation Price MAE')
plt.title('MAE over epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()


image_paths = [os.path.abspath('test/p1.png'),
               os.path.abspath('test/p2.png'),
               os.path.abspath('test/p3.png')]

# Extract features for all images
image_features = [image_processing.get_image_features(img_path) for img_path in image_paths]
# Location data for Cluj Napoca (same location for all images)
cluj_lat = 46.7712  # Latitude for Cluj Napoca
cluj_lon = 23.6236  # Longitude for Cluj Napoca # Repeat for each image

img = np.mean(image_features, axis=0).reshape(1, -1)  # Shape (1, 2048)
location_data = np.array([cluj_lat, cluj_lon], dtype='float32').reshape(1, -1)  # Shape (1, 2)

# Make predictions with the trained model
predictions = model.predict([img, location_data])
print(predictions)
# # Display predictions
# for i, pred in enumerate(predictions):
#     print(f"Predictions for image {i+1}:")
#     print(f"Room Count: {pred[0]}")
#     print(f"Kitchens: {pred[1]}")
#     print(f"Bathrooms: {pred[2]}")
#     print(f"Estimated Price: {pred[3]} EUR\n")


#if __name__ == "__main__":