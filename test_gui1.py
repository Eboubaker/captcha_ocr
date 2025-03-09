import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from lib import predict_captcha, preprocess_image, CAPTCHA_LENGTH, CHARACTERS

# Define constants (same as in your training script)
IMG_HEIGHT = 35
IMG_WIDTH = 150

class CaptchaRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = None
        self.loadModel()
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle('Captcha Recognition')
        self.setGeometry(100, 100, 800, 500)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create button for selecting an image
        self.select_btn = QPushButton('Select Captcha Image')
        self.select_btn.clicked.connect(self.selectImage)
        main_layout.addWidget(self.select_btn)
        
        # Create layout for images
        images_layout = QHBoxLayout()
        
        # Original image
        original_layout = QVBoxLayout()
        original_layout.addWidget(QLabel('Original Image:'))
        self.original_img_label = QLabel()
        self.original_img_label.setAlignment(Qt.AlignCenter)
        self.original_img_label.setMinimumSize(200, 100)
        original_layout.addWidget(self.original_img_label)
        images_layout.addLayout(original_layout)
        
        # Preprocessed image
        preprocessed_layout = QVBoxLayout()
        preprocessed_layout.addWidget(QLabel('Preprocessed Image:'))
        self.preprocessed_img_label = QLabel()
        self.preprocessed_img_label.setAlignment(Qt.AlignCenter)
        self.preprocessed_img_label.setMinimumSize(200, 100)
        preprocessed_layout.addWidget(self.preprocessed_img_label)
        images_layout.addLayout(preprocessed_layout)
        
        main_layout.addLayout(images_layout)
        
        # Create label for prediction result
        result_layout = QVBoxLayout()
        result_layout.addWidget(QLabel('Predicted Captcha:'))
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        result_layout.addWidget(self.result_label)
        main_layout.addLayout(result_layout)
        
        # Status label
        self.status_label = QLabel('Please select a captcha image')
        main_layout.addWidget(self.status_label)
    
    def loadModel(self):
        """Load the trained captcha model"""
        model_path = "captcha_model.keras"
        if os.path.exists(model_path):
            try:
                self.status_label.setText("Loading model...")
                QApplication.processEvents()
                self.model = models.load_model(model_path)
                self.status_label.setText("Model loaded successfully")
            except Exception as e:
                self.status_label.setText(f"Error loading model: {str(e)}")
        else:
            self.status_label.setText(f"Model file '{model_path}' not found!")
    
    def selectImage(self):
        """Open file dialog to select an image"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Captcha Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", 
            options=options
        )
        
        if file_path:
            self.processImage(file_path)
    
    def processImage(self, image_path):
        """Process the selected image and predict captcha"""
        # Load and display original image
        original_img = cv2.imread(image_path)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        self.displayImage(original_img_rgb, self.original_img_label)
        
        # Preprocess image and display
        preprocessed_img_display = preprocess_image(image_path)
        # Resize for better visibility
        preprocessed_img_display = cv2.resize(preprocessed_img_display, (IMG_WIDTH * 2, IMG_HEIGHT * 2))
        self.displayImage(preprocessed_img_display, self.preprocessed_img_label, is_grayscale=True)
        
        # Predict captcha
        if self.model is not None:
            captcha_text = predict_captcha(self.model, image_path)
            self.result_label.setText(captcha_text)
            self.status_label.setText(f"Prediction complete: {captcha_text}")
        else:
            self.status_label.setText("Error: Model not loaded")
                
    
    def displayImage(self, img, label, is_grayscale=False):
        """Display an image on a QLabel"""
        h, w = img.shape[:2]
        
        if is_grayscale:
            # If grayscale, create RGB image from it
            q_img = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            bytes_per_line = 3 * w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

def main():
    app = QApplication(sys.argv)
    window = CaptchaRecognitionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()