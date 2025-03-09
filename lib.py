import cv2
import numpy as np
from pyparsing import C

CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IMG_HEIGHT = 35
IMG_WIDTH = 150
CAPTCHA_LENGTH = 5
CHAR_CANVAS_SIZE = 35
def predict_character(model, char_img):
    """Predict a single character from a segmented character image.
    
    Args:
        model: The trained model for character recognition
        char_img: Preprocessed and segmented character image
        
    Returns:
        Predicted character
    """
    # Reshape for model input (add batch and channel dimensions)
    img_input = char_img.reshape(1, char_img.shape[0], char_img.shape[1], 1)
    
    # Make prediction
    pred = model.predict(img_input, verbose=0)
    
    # Get the character with highest probability
    char_idx = np.argmax(pred[0])
    
    return CHARACTERS[char_idx]

def predict_captcha(model, image_path):
    """Predict the text in a CAPTCHA image using the trained model.
    
    Args:
        model: The trained model for character recognition.
        image_path: Path to the CAPTCHA image file.
        
    Returns:
        String with the predicted CAPTCHA text.
    """
    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)
    
    # Segment the image into individual characters
    character_images = segment_characters(preprocessed_img)

    # If no characters were found, return an empty string
    if not character_images:
        return ""
    
    # Predict each character using the predict_character function
    result = "".join(predict_character(model, char_img) for char_img in character_images)
    
    return result

def preprocess_image(image_path):
    """Preprocess a single image."""
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to standard dimensions
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Apply strong binary thresholding (similar to -threshold 80%)
    _, img = cv2.threshold(img, 204, 255, cv2.THRESH_BINARY)  # 204 is ~80% of 255
    
    # Create kernels for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    # Apply dilation (similar to -morphology Dilate Rectangle:2x2)
    img = cv2.dilate(img, kernel, iterations=1)
    
    # Apply erosion (similar to -morphology Erode Rectangle:2x2)
    img = cv2.erode(img, kernel, iterations=1)
    
    # Normalize
    img = img / 255.0
    
    return img

def segment_characters(preprocessed_img, padding=2, canvas_size=(35, 35)):
    """
    Segments characters from a preprocessed CAPTCHA image and centers them in a fixed-size canvas
    without distortion.

    Args:
        preprocessed_img (numpy.ndarray): Preprocessed grayscale image.
        padding (int): Number of pixels to add as padding around each character.
        canvas_size (tuple): Size of the canvas (width, height) to place each character.

    Returns:
        List of numpy.ndarray: List of 35x35 character images with the same scale.
    """
    if preprocessed_img is None:
        raise ValueError("Invalid image input")

    # Convert image back to 0-255 range if normalized
    if preprocessed_img.max() <= 1:
        preprocessed_img = (preprocessed_img * 255).astype(np.uint8)

    # Apply thresholding to ensure binary image
    _, thresh = cv2.threshold(preprocessed_img, 127, 255, cv2.THRESH_BINARY_INV)

    # Create a copy of the thresholded image for removing small objects
    cleaned_thresh = thresh.copy()
    
    # Find small noise contours first (without dilation)
    small_contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify and remove small noise objects
    for contour in small_contours:
        if cv2.contourArea(contour) < 10:
            # Fill the small contour area with black (0) to remove the noise
            cv2.drawContours(cleaned_thresh, [contour], 0, 0, -1)
    thresh = cleaned_thresh
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours of the characters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Remove small noise by filtering out contours smaller than 5x5
    contours = [c for c in contours if cv2.boundingRect(c)[2] >= 5 and cv2.boundingRect(c)[3] >= 8]
    # contours = [c for c in contours if cv2.contourArea(c) >= 80]

    # Sort contours from left to right based on x-coordinates
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Extract segmented characters and center them in a 35x35 white canvas
    character_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Apply padding while ensuring we stay within image bounds
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(preprocessed_img.shape[1] - x, w + 2 * padding)
        h = min(preprocessed_img.shape[0] - y, h + 2 * padding)

        # Extract character with padding
        char_crop = preprocessed_img[y:y + h, x:x + w]

        # Ensure the character fits inside 35x35
        if w > canvas_size[0] or h > canvas_size[1]:
            scale_factor = min(canvas_size[0] / w, canvas_size[1] / h)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            char_crop = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_w, new_h = w, h

        # Create a white canvas
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255

        # Calculate the position to center the character in the canvas
        x_offset = (canvas_size[0] - new_w) // 2
        y_offset = (canvas_size[1] - new_h) // 2

        # Place the character in the center
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = char_crop

        character_images.append(canvas)

    return character_images

def segment_characters_detailed(preprocessed_img, padding=2, canvas_size=(35, 35)):
    """
    Segments characters from a preprocessed CAPTCHA image and centers them in a fixed-size canvas
    without distortion. First removes small noise objects before character extraction.
    
    Args:
        preprocessed_img (numpy.ndarray): Preprocessed grayscale image.
        padding (int): Number of pixels to add as padding around each character.
        canvas_size (tuple): Size of the canvas (width, height) to place each character.
    
    Returns:
        List of numpy.ndarray: List of 35x35 character images with the same scale.
        List of tuple: Bounding rectangles for visualization.
        numpy.ndarray: Processed binary image after thresholding.
    """
    if preprocessed_img is None:
        raise ValueError("Invalid image input")
    
    # Convert image back to 0-255 range if normalized
    if preprocessed_img.max() <= 1:
        preprocessed_img = (preprocessed_img * 255).astype(np.uint8)
    
    # Apply thresholding to ensure binary image
    _, thresh = cv2.threshold(preprocessed_img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Create a copy of the thresholded image for removing small objects
    cleaned_thresh = thresh.copy()
    
    # Find small noise contours first (without dilation)
    small_contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify and remove small noise objects
    for contour in small_contours:
        if cv2.contourArea(contour) < 10:
            # Fill the small contour area with black (0) to remove the noise
            cv2.drawContours(cleaned_thresh, [contour], 0, 0, -1)
    
    # Now apply dilation to the cleaned image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_thresh = cv2.dilate(cleaned_thresh, kernel, iterations=1)
    
    # Find contours of the characters on the dilated image
    contours, _ = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out any remaining small contours
    contours = [c for c in contours if cv2.contourArea(c) >= 30]
    
    # Sort contours from left to right based on x-coordinates
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # Store bounding rectangles for visualization
    bounding_rects = [cv2.boundingRect(c) for c in contours]
    
    # Extract segmented characters and center them in a 35x35 white canvas
    character_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Apply padding while ensuring we stay within image bounds
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(preprocessed_img.shape[1] - x, w + 2 * padding)
        h = min(preprocessed_img.shape[0] - y, h + 2 * padding)
        
        # Extract character with padding
        char_crop = preprocessed_img[y:y + h, x:x + w]
        
        # Ensure the character fits inside 35x35
        if w > canvas_size[0] or h > canvas_size[1]:
            scale_factor = min(canvas_size[0] / w, canvas_size[1] / h)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            char_crop = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_w, new_h = w, h
            
        # Create a white canvas
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255
        
        # Calculate the position to center the character in the canvas
        x_offset = (canvas_size[0] - new_w) // 2
        y_offset = (canvas_size[1] - new_h) // 2
        
        # Place the character in the center
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = char_crop
        character_images.append(canvas)
        
    return character_images, bounding_rects, dilated_thresh
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    # Load the image
    image_path = "labeled/sLqhd.png"
    
    # Get original image
    original_img = cv2.imread(image_path)
    # Convert from BGR to RGB for matplotlib
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Load the trained model
    try:
        from tensorflow.keras.models import load_model
        model = load_model("captcha_model.keras")
        preprocessed_img = preprocess_image(image_path)
        segmented_chars, bounding_rects, thresh_img = segment_characters_detailed(preprocessed_img)
        char_predictions = []
        for char in segmented_chars:
            char_predictions.append(predict_character(model, char))
        # Predict the CAPTCHA and get segmentation results
        predicted_text = predict_captcha(model, image_path)
        model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")
        # If model can't be loaded, use the previous approach
        predicted_text = "N/A (Model not loaded)"
        char_predictions = ["N/A"] * len(segmented_chars)
        model_loaded = False
    
    # Create a copy of the original image for drawing bounding boxes
    img_with_boxes = original_img_rgb.copy()
    
    # Draw red boxes around detected characters
    for x, y, w, h in bounding_rects:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    # Get filename without extension for display
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Set up a single figure for all visualizations
    n_chars = len(segmented_chars)
    
    # Calculate grid dimensions based on number of characters
    n_rows = 3  # Fixed rows: original+processed images, threshold+boxes, segmented chars
    n_cols = max(4, n_chars)  # At least 4 columns, or more if needed for chars
    
    # Create the figure
    plt.figure(figsize=(15, 10))
    
    # Row 1: Original and Preprocessed Images
    plt.subplot(n_rows, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_img_rgb)
    plt.axis('off')
    
    plt.subplot(n_rows, 2, 2)
    plt.title('Preprocessed Image')
    plt.imshow(preprocessed_img, cmap='gray')
    plt.axis('off')
    
    # Row 2: Threshold Image and Image with Boxes
    plt.subplot(n_rows, 2, 3)
    plt.title('Threshold Image for Contour Detection')
    plt.imshow(thresh_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(n_rows, 2, 4)
    plt.title(f'Detected Characters with Prediction: {predicted_text}')
    plt.imshow(img_with_boxes)
    plt.axis('off')
    
    # Row 3: Segmented Characters with Predictions
    if n_chars > 0:
        for i, char_img in enumerate(segmented_chars):
            plt.subplot(n_rows, n_chars, (2 * n_chars) + i + 1)
            plt.imshow(char_img, cmap='gray')
            if model_loaded:
                plt.title(f'Pred: {char_predictions[i]}')
            else:
                plt.title(f'Char {i+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"CAPTCHA Filename: {filename}, Predicted: {predicted_text}", fontsize=16)
    plt.subplots_adjust(top=0.9)  # Adjust to make room for the suptitle
    
    plt.show()
    
    print(f"CAPTCHA filename: {filename}")
    print(f"Number of characters detected: {n_chars}")
    print(f"Predicted text: {predicted_text}")