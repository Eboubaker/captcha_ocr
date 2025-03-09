import os
import sys
import numpy as np
from silence_tensorflow import silence_tensorflow

silence_tensorflow("NONE")

import tensorflow as tf
from lib import CAPTCHA_LENGTH, CHARACTERS, preprocess_image, segment_characters


def predict_captcha(model, image_path):
    """Predict the text in a CAPTCHA image."""
    try:
        # Preprocess the image
        preprocessed_img = preprocess_image(image_path)
        
        # Segment the image into individual characters
        character_images = segment_characters(preprocessed_img)
        if len(character_images) != CAPTCHA_LENGTH:
            print(f"error: invalid segment length={len(character_images)}")       
        result = ""
        # Process each character
        for char_img in character_images:
            # Reshape for model input (add batch and channel dimensions)
            img_input = char_img.reshape(1, char_img.shape[0], char_img.shape[1], 1)
            
            # Make prediction
            pred = model.predict(img_input, verbose=0)
            
            # Get the character with highest probability
            char_idx = np.argmax(pred[0])
            
            # Add to result
            result += CHARACTERS[char_idx]
        
        return result
    except Exception as e:
        raise Exception(f"error: {str(e)}")

def main():
    try:
        # Check if an image path was provided
        if len(sys.argv) != 2:
            print("error: Please provide a CAPTCHA image path")
            print("Usage: python predict_captcha.py <image_path>")
            return 1
        
        image_path = sys.argv[1]
        
        # Check if the file exists
        if not os.path.isfile(image_path):
            print(f"error: File not found: {image_path}")
            return 1
        
        # Load the model
        model_path = "captcha_model.keras"
        if not os.path.isfile(model_path):
            print(f"error: Model file not found: {model_path}")
            return 1
        
        model = tf.keras.models.load_model(model_path)
        
        # Predict the CAPTCHA
        predicted_text = predict_captcha(model, image_path)
        
        # Print the result
        print(predicted_text)
        return 0
        
    except Exception as e:
        print(f"error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())