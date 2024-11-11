import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QScrollArea
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from keras.src.saving import load_model
from image_processing import ImageProcessing
from location_processing import LocationProcessor


class PropertyEvaluationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_processing = ImageProcessing()
        self.location_processing = LocationProcessor()
        self.setWindowTitle("Property Evaluation Demo")
        self.setGeometry(100, 100, 800, 600)
        self.model = load_model('best_model.keras')  # load your pre-trained model
        self.layout = QVBoxLayout()

        self.setStyleSheet("background-color: #f0f0f0;")

        # Location input
        self.location_label = QLabel("Enter Property Location:")
        self.location_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.location_label.setStyleSheet("color: black;")
        self.layout.addWidget(self.location_label)

        self.location_input = QLineEdit()
        self.location_input.setStyleSheet("""
            padding: 10px;
            font-size: 14px;
            border-radius: 5px;
            border: 1px solid #ccc;
            background-color: white;
            color: black;
            font-family: Arial, sans-serif;
        """)
        self.layout.addWidget(self.location_input)

        # image selection button
        self.select_button = QPushButton("Select Property Images")
        self.select_button.setStyleSheet("""
            padding: 12px 24px;
            font-size: 14px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
        """)
        self.select_button.clicked.connect(self.select_images)
        self.layout.addWidget(self.select_button)

        # scrollable area for image previews
        self.image_preview_scroll_area = QScrollArea()
        self.image_preview_scroll_area.setWidgetResizable(True)
        self.image_preview_widget = QWidget()
        self.image_preview_layout = QVBoxLayout(self.image_preview_widget)
        self.image_preview_scroll_area.setWidget(self.image_preview_widget)
        self.layout.addWidget(self.image_preview_scroll_area)

        # Scrollable area for image paths
        self.image_paths_scroll_area = QScrollArea()
        self.image_paths_scroll_area.setWidgetResizable(True)
        self.image_paths_widget = QWidget()
        self.image_paths_layout = QVBoxLayout(self.image_paths_widget)
        self.image_paths_scroll_area.setWidget(self.image_paths_widget)
        self.layout.addWidget(self.image_paths_scroll_area)

        # label for image paths
        self.images_label = QLabel()
        font = self.images_label.font()
        font.setItalic(True)
        self.images_label.setFont(font)
        self.images_label.setStyleSheet("color: #888;")
        self.layout.addWidget(self.images_label)

        # evaluate button
        self.evaluate_button = QPushButton("Evaluate Property")
        self.evaluate_button.setStyleSheet("""
            padding: 12px 24px;
            font-size: 14px;
            background-color: #28a745;
            color: white;
            border: none;
            border-radius: 5px;
        """)
        self.evaluate_button.clicked.connect(self.evaluate_property)
        self.layout.addWidget(self.evaluate_button)

        # results display
        self.results_label = QLabel("Results will be shown here.")
        self.results_label.setFont(QFont("Arial", 12))
        self.results_label.setStyleSheet("color: #333;")
        self.layout.addWidget(self.results_label)

        self.setLayout(self.layout)

        # initialize the list of selected images and paths
        self.selected_images = []
        self.selected_paths = []

    def select_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Property Images", "", "Images (*.png *.jpg *.jpeg)")
        if file_paths:
            file_paths = file_paths[:3]
            self.selected_images.extend(file_paths)
            self.update_image_previews()
            self.update_image_paths()

    def update_image_previews(self):
        for i in reversed(range(self.image_preview_layout.count())):
            widget = self.image_preview_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        for image_path in self.selected_images:
            label = QLabel(self)
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            self.image_preview_layout.addWidget(label)

    def update_image_paths(self):
        for i in reversed(range(self.image_paths_layout.count())):
            widget = self.image_paths_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        for path in self.selected_images:
            path_label = QLabel(path)
            path_label.setStyleSheet("""
                font-size: 12px;
                color: #555;
                margin-bottom: 5px;
            """)
            self.image_paths_layout.addWidget(path_label)

    def evaluate_property(self):
        location = self.location_input.text()

        # Check if location is provided and at least one image is selected
        if not location or len(self.selected_images) == 0:
            self.results_label.setText("Please select images and provide a location.")
            return


        image_features = [self.image_processing.get_image_features(img_path) for img_path in self.selected_images]
        img = np.mean(image_features, axis=0).reshape(1, -1)

        lat, lon = self.location_processing.get_lat_lon(location)
        location_data = np.array([lat, lon], dtype='float32').reshape(1, -1)

        location_data = self.location_processing.scaler.fit_transform(location_data)

        predictions = self.model.predict([img, location_data])

        mean_value = 500000
        std_value = 200000

        # handle predictions [room_count, kitchens, bathrooms, normalized_value]
        predicted_room_count = predictions[0][0][0]
        predicted_kitchens = predictions[1][0][0]
        predicted_bathrooms = predictions[2][0][0]
        predicted_value_normalized = predictions[3][0][0]  # normalized value predicted by the model

        predicted_room_count = round(predicted_room_count)
        predicted_kitchens = round(predicted_kitchens)
        predicted_bathrooms = round(predicted_bathrooms)

        # denormalize the price (property value)
        predicted_value = (predicted_value_normalized * std_value) + mean_value

        # display the evaluation results
        self.results_label.setText(f"Evaluating property at {location} with {len(self.selected_images)} images.\n"
                                   f"Predicted Room Count: {predicted_room_count:.2f}\n"
                                   f"Predicted Kitchens: {predicted_kitchens:.2f}\n"
                                   f"Predicted Bathrooms: {predicted_bathrooms:.2f}\n"
                                   f"Predicted Property Value: {predicted_value:.2f}€")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PropertyEvaluationApp()
    window.show()
    sys.exit(app.exec_())
