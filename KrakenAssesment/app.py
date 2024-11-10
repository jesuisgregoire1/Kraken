import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QHBoxLayout, QScrollArea, QFrame, QVBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt


class PropertyEvaluationApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Property Evaluation Demo")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        self.layout = QVBoxLayout()

        # Set window background color
        self.setStyleSheet("background-color: #f0f0f0;")

        # Location input
        self.location_label = QLabel("Enter Property Location:")
        self.location_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.location_label.setStyleSheet("color: #333;")
        self.layout.addWidget(self.location_label)

        self.location_input = QLineEdit()
        self.location_input.setStyleSheet("""
            padding: 10px;
            font-size: 14px;
            border-radius: 5px;
            border: 1px solid #ccc;
            background-color: white;
        """)
        self.layout.addWidget(self.location_input)

        # Image selection button
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

        # Scrollable area for image previews
        self.image_preview_scroll_area = QScrollArea()
        self.image_preview_scroll_area.setWidgetResizable(True)
        self.image_preview_widget = QWidget()
        self.image_preview_layout = QVBoxLayout(self.image_preview_widget)  # Changed to QVBoxLayout for vertical scrolling
        self.image_preview_scroll_area.setWidget(self.image_preview_widget)
        self.layout.addWidget(self.image_preview_scroll_area)

        # Scrollable area for image paths
        self.image_paths_scroll_area = QScrollArea()
        self.image_paths_scroll_area.setWidgetResizable(True)
        self.image_paths_widget = QWidget()
        self.image_paths_layout = QVBoxLayout(self.image_paths_widget)  # Changed to QVBoxLayout for vertical scrolling
        self.image_paths_scroll_area.setWidget(self.image_paths_widget)
        self.layout.addWidget(self.image_paths_scroll_area)

        # Label for image paths
        self.images_label = QLabel("No images selected.")
        font = self.images_label.font()  # Get the current font
        font.setItalic(True)  # Set the font to italic
        self.images_label.setFont(font)  # Apply the modified font to the label
        self.images_label.setStyleSheet("color: #888;")
        self.layout.addWidget(self.images_label)

        # Evaluate button
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

        # Results display
        self.results_label = QLabel("Results will be shown here.")
        self.results_label.setFont(QFont("Arial", 12))
        self.results_label.setStyleSheet("color: #333;")
        self.layout.addWidget(self.results_label)

        self.setLayout(self.layout)

        # Initialize the list of selected images and paths
        self.selected_images = []
        self.selected_paths = []

    def select_images(self):
        # Open a file dialog to select multiple images
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Property Images", "", "Images (*.png *.jpg *.jpeg)")
        if file_paths:
            # Limit the number of selected images to 3
            file_paths = file_paths[:3]

            # Add the selected images to the list
            self.selected_images.extend(file_paths)

            # Clear the current image previews and paths
            self.update_image_previews()
            self.update_image_paths()

            # Check if the images exceed 3, and display scroll area if needed
            if len(self.selected_images) >= 3:
                self.image_preview_scroll_area.setVisible(True)

    def update_image_previews(self):
        # Clear the current layout items (for both images and paths)
        for i in reversed(range(self.image_preview_layout.count())):
            widget = self.image_preview_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Add the selected images to the preview area
        for image_path in self.selected_images:
            label = QLabel(self)
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio)  # Resize image to fit
            label.setPixmap(pixmap)
            self.image_preview_layout.addWidget(label)

        # Hide scroll area initially until 3 images are selected
        if len(self.selected_images) < 3:
            self.image_preview_scroll_area.setVisible(False)

    def update_image_paths(self):
        # Clear the paths layout first
        for i in reversed(range(self.image_paths_layout.count())):
            widget = self.image_paths_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Add the paths to the paths section
        for path in self.selected_images:
            path_label = QLabel(path)
            path_label.setStyleSheet("""
                font-size: 12px;
                color: #555;
                margin-bottom: 5px;
            """)
            self.image_paths_layout.addWidget(path_label)

        # Hide scroll area initially until 3 paths are selected
        if len(self.selected_images) < 3:
            self.image_paths_scroll_area.setVisible(False)

    def evaluate_property(self):
        location = self.location_input.text()
        if not location or len(self.selected_images) == 0:
            self.results_label.setText("Please select images and provide a location.")
        else:
            self.results_label.setText(f"Evaluating property at {location} with {len(self.selected_images)} images.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PropertyEvaluationApp()
    window.show()
    sys.exit(app.exec_())
