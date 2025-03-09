import os
import sys
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import shutil
from silence_tensorflow import silence_tensorflow

silence_tensorflow("NONE")

import tensorflow as tf
from lib import CAPTCHA_LENGTH, CHARACTERS, preprocess_image, segment_characters

class CaptchaReviewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Captcha Reviewer")
        self.root.geometry("800x600")
        
        # Store paths
        self.current_image_path = None
        self.input_folder = None
        self.corrections_folder = None
        self.valid_folder = None
        self.incorrect_segments_folder = None
        self.image_files = []
        self.current_index = 0
        self.current_prediction = ""
        self.current_segments_count = 0
        
        # Model loading
        self.model = None
        self.load_model()
        
        # Create GUI components
        self.create_widgets()
    
    def load_model(self):
        """Load the TensorFlow model."""
        try:
            model_path = "captcha_model.keras"
            if not os.path.isfile(model_path):
                print(f"Error: Model file not found: {model_path}")
                self.model = None
            else:
                self.model = tf.keras.models.load_model(model_path)
                print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
    
    def create_widgets(self):
        """Create all GUI components."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Folder selection buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(btn_frame, text="Select Folders", command=self.select_folders).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Next Image", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Folder display
        folders_frame = ttk.Frame(main_frame)
        folders_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.folders_label = ttk.Label(folders_frame, text="No folders selected", wraplength=780)
        self.folders_label.pack(anchor=tk.W)
        
        # Segmented characters display
        self.segmented_label = ttk.Label(main_frame, text="Segmented Characters:")
        self.segmented_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.segmented_image_label = ttk.Label(main_frame)
        self.segmented_image_label.pack(fill=tk.X, pady=(0, 10))
        
        # Captcha image display
        self.image_label = ttk.Label(main_frame, text="Captcha Image:")
        self.image_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.captcha_image_label = ttk.Label(main_frame)
        self.captcha_image_label.pack(fill=tk.X, pady=(0, 10))
        
        # Prediction display
        self.prediction_label = ttk.Label(main_frame, text="Model Prediction:")
        self.prediction_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.prediction_text = tk.Text(main_frame, height=2, width=20, font=("Arial", 20))
        self.prediction_text.pack(fill=tk.X, pady=(0, 10))
        self.prediction_text.config(state=tk.DISABLED)
        
        # Correction input
        self.correction_label = ttk.Label(main_frame, text="Correction (if needed):")
        self.correction_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.correction_entry = ttk.Entry(main_frame, font=("Arial", 20))
        self.correction_entry.pack(fill=tk.X, pady=(0, 10))
        self.correction_entry.bind("<Return>", self.process_correction)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please select folders to begin")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_folders(self):
        """Let user select input and output folders."""
        # Select input folder
        self.input_folder = filedialog.askdirectory(title="Select folder with captcha images")
        if not self.input_folder:
            return
        
        # Select corrections folder (for manually corrected images)
        self.corrections_folder = filedialog.askdirectory(title="Select folder for corrected images")
        if not self.corrections_folder:
            return
        
        # Select valid folder (for automatically validating predictions)
        self.valid_folder = filedialog.askdirectory(title="Select folder for valid predictions")
        if not self.valid_folder:
            return
        
        # Select incorrect segments folder
        self.incorrect_segments_folder = filedialog.askdirectory(title="Select folder for incorrect segments count")
        if not self.incorrect_segments_folder:
            return
        
        # Update folders display
        folder_text = (
            f"Input: {self.input_folder}\n"
            f"Corrections: {self.corrections_folder}\n"
            f"Valid: {self.valid_folder}\n"
            f"Incorrect Segments: {self.incorrect_segments_folder}"
        )
        self.folders_label.config(text=folder_text)
        
        # Get all image files
        self.image_files = [
            os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        self.current_index = 0
        self.status_var.set(f"Found {len(self.image_files)} images")
        
        if self.image_files:
            self.load_current_image()
    
    def load_current_image(self):
        """Load and process the current image."""
        if self.current_index >= len(self.image_files):
            self.status_var.set("No more images to process")
            return
        
        self.current_image_path = self.image_files[self.current_index]
        file_name = os.path.basename(self.current_image_path)
        
        # Show image path
        self.status_var.set(f"Image {self.current_index + 1}/{len(self.image_files)}: {file_name}")
        
        try:
            # Display the original image
            img = Image.open(self.current_image_path)
            img = img.resize((400, 100), Image.LANCZOS)  # Resize for display
            img_tk = ImageTk.PhotoImage(img)
            self.captcha_image_label.config(image=img_tk)
            self.captcha_image_label.image = img_tk  # Keep a reference
            
            # Process the image
            self.process_image(self.current_image_path)
            
            # Clear correction entry
            self.correction_entry.delete(0, tk.END)
            self.correction_entry.focus()
        
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
    
    def process_image(self, image_path):
        """Process the image, predict text, and show segmented characters."""
        try:
            # Preprocess the image
            preprocessed_img = preprocess_image(image_path)
            
            # Segment characters and add red borders
            character_images = segment_characters(preprocessed_img)
            self.current_segments_count = len(character_images)
            bordered_segments = []
            
            for segment in character_images:
                # Add red border to each segment
                bordered = cv2.copyMakeBorder(
                    segment, 2, 2, 2, 2, 
                    cv2.BORDER_CONSTANT, 
                    value=[0, 0, 255]  # Red color in BGR
                )
                
                # Convert to RGB for display
                if bordered.ndim == 2:  # If grayscale
                    bordered = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
                
                bordered_segments.append(bordered)
            
            # Concatenate for display
            if bordered_segments:
                segmented_display = cv2.hconcat(bordered_segments)
                
                # Convert to PIL image for Tkinter
                seg_img = Image.fromarray(cv2.cvtColor(segmented_display, cv2.COLOR_BGR2RGB))
                seg_img = seg_img.resize((400, 100), Image.LANCZOS)  # Resize for display
                seg_img_tk = ImageTk.PhotoImage(seg_img)
                self.segmented_image_label.config(image=seg_img_tk)
                self.segmented_image_label.image = seg_img_tk  # Keep a reference
            
            # Update segment status
            if self.current_segments_count != CAPTCHA_LENGTH:
                self.segmented_label.config(text=f"Segmented Characters (INVALID COUNT: {self.current_segments_count}/5):")
            else:
                self.segmented_label.config(text="Segmented Characters (5/5):")
            
            # Predict if model is available
            if self.model:
                self.current_prediction = self.predict_captcha(image_path)
                
                # Update prediction text
                self.prediction_text.config(state=tk.NORMAL)
                self.prediction_text.delete(1.0, tk.END)
                self.prediction_text.insert(tk.END, self.current_prediction)
                self.prediction_text.config(state=tk.DISABLED)
            else:
                self.status_var.set("Model not loaded - cannot predict")
                self.current_prediction = ""
        
        except Exception as e:
            self.status_var.set(f"Error processing image: {str(e)}")
            self.current_prediction = ""
            self.current_segments_count = 0
    
    def predict_captcha(self, image_path):
        """Predict the text in a CAPTCHA image."""
        try:
            # Preprocess the image
            preprocessed_img = preprocess_image(image_path)
            
            # Segment the image into individual characters
            character_images = segment_characters(preprocessed_img)
            if len(character_images) != 5:
                return f"Invalid:{len(character_images)}"
                
            result = ""
            # Process each character
            for char_img in character_images:
                # Reshape for model input (add batch and channel dimensions)
                img_input = char_img.reshape(1, char_img.shape[0], char_img.shape[1], 1)
                
                # Make prediction
                pred = self.model.predict(img_input, verbose=0)
                
                # Get the character with highest probability
                char_idx = np.argmax(pred[0])
                
                # Add to result
                result += CHARACTERS[char_idx]
            
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def process_correction(self, event=None):
        """Process the user's correction and move to the next image."""
        if not self.current_image_path:
            return
        
        # Check if all required folders are selected
        if not all([self.corrections_folder, self.valid_folder, self.incorrect_segments_folder]):
            self.status_var.set("Error: Please select all folders first")
            return
        
        correction = self.correction_entry.get().strip()
        base_name = os.path.basename(self.current_image_path)
        file_ext = os.path.splitext(base_name)[1]
        
        try:
            # Determine destination based on segment count and correction input
            if self.current_segments_count != 5:
                # Image has incorrect segment count - move to incorrect_segments folder
                if correction:
                    # User provided correction text
                    new_file_path = os.path.join(self.incorrect_segments_folder, correction + file_ext)
                    shutil.move(self.current_image_path, new_file_path)
                    self.status_var.set(f"Invalid segments ({self.current_segments_count}/5): Saved as '{correction}{file_ext}' in incorrect segments folder")
                else:
                    # No correction provided, still move to incorrect_segments with original name
                    new_file_path = os.path.join(self.incorrect_segments_folder, base_name)
                    shutil.move(self.current_image_path, new_file_path)
                    self.status_var.set(f"Invalid segments ({self.current_segments_count}/5): Saved with original name in incorrect segments folder")
            else:
                # Image has correct segment count (5)
                if correction:
                    # User provided correction - move to corrections folder
                    new_file_path = os.path.join(self.corrections_folder, correction + file_ext)
                    shutil.move(self.current_image_path, new_file_path)
                    self.status_var.set(f"Corrected: Saved as '{correction}{file_ext}' in corrections folder")
                else:
                    # No correction - move to valid folder with prediction as name
                    new_file_path = os.path.join(self.valid_folder, self.current_prediction + file_ext)
                    shutil.move(self.current_image_path, new_file_path)
                    self.status_var.set(f"Valid: Saved as '{self.current_prediction}{file_ext}' in valid folder")
        
        except Exception as e:
            self.status_var.set(f"Error saving file: {str(e)}")
        
        # Move to the next image
        self.next_image()
    
    def next_image(self):
        """Load the next image."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
        else:
            self.status_var.set("All images have been processed")

if __name__ == "__main__":
    root = tk.Tk()
    app = CaptchaReviewerApp(root)
    root.mainloop()