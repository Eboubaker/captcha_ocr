import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk

from lib import predict_captcha, preprocess_image, CHARACTERS

# Define constants (same as in your training script)
IMG_HEIGHT = 35
IMG_WIDTH = 150

class CaptchaRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Captcha Recognition")
        self.root.geometry("800x500")
        
        self.model = None
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        # Main frame
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Button to select image
        select_btn = Button(main_frame, text="Select Captcha Image", command=self.select_image)
        select_btn.pack(pady=10)
        
        # Frame for images
        images_frame = Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image frame
        original_frame = Frame(images_frame)
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        Label(original_frame, text="Original Image:").pack()
        self.original_img_label = Label(original_frame, bg="lightgray")
        self.original_img_label.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Preprocessed image frame
        preprocessed_frame = Frame(images_frame)
        preprocessed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        Label(preprocessed_frame, text="Preprocessed Image:").pack()
        self.preprocessed_img_label = Label(preprocessed_frame, bg="lightgray")
        self.preprocessed_img_label.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Prediction result frame
        result_frame = Frame(main_frame)
        result_frame.pack(fill=tk.X, pady=10)
        Label(result_frame, text="Predicted Captcha:").pack()
        self.result_label = Label(result_frame, text="", font=("Arial", 24, "bold"))
        self.result_label.pack(pady=5)
        
        # Status bar
        self.status_label = Label(main_frame, text="Please select a captcha image", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model(self):
        """Load the trained captcha model"""
        model_path = "captcha_model.h5"
        if os.path.exists(model_path):
            try:
                self.status_label.config(text="Loading model...")
                self.root.update()
                self.model = models.load_model(model_path)
                self.status_label.config(text="Model loaded successfully")
            except Exception as e:
                self.status_label.config(text=f"Error loading model: {str(e)}")
        else:
            self.status_label.config(text=f"Model file '{model_path}' not found!")
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Captcha Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.process_image(file_path)
    
    def process_image(self, image_path):
        """Process the selected image and predict captcha"""
        # Load and display original image
        original_img = cv2.imread(image_path)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        self.display_image(original_img_rgb, self.original_img_label)
        
        # Preprocess image and display
        preprocessed_img_display = preprocess_image(image_path)
        # Resize for better visibility
        preprocessed_img_display = cv2.resize(preprocessed_img_display, (IMG_WIDTH * 2, IMG_HEIGHT * 2))
        self.display_image(preprocessed_img_display, self.preprocessed_img_label, is_grayscale=True)
        
        # Predict captcha
        if self.model is not None:
            captcha_text = predict_captcha(self.model, image_path)
            self.result_label.config(text=captcha_text)
            self.status_label.config(text=f"Prediction complete: {captcha_text}")
        else:
            self.status_label.config(text="Error: Model not loaded")
            
    
    def display_image(self, img, label, is_grayscale=False):
        """Display an image on a Tkinter Label"""
        if is_grayscale:
            # For grayscale images
            pil_img = Image.fromarray(img)
        else:
            # For RGB images
            pil_img = Image.fromarray(img)
        
        # Resize image to fit in the label while maintaining aspect ratio
        width, height = pil_img.size
        max_size = 300  # Maximum dimension
        scale = min(max_size/width, max_size/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        tk_img = ImageTk.PhotoImage(pil_img)
        # Keep a reference to prevent garbage collection
        label.image = tk_img
        label.config(image=tk_img)

def main():
    root = tk.Tk()
    app = CaptchaRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()