import os
import sys
import cv2
import numpy as np
from lib import CHARACTERS, preprocess_image, segment_characters

def main():
    try:
        image_path = "correct/04804.png"
        preprocessed_image = preprocess_image(image_path)

        # Segment characters
        segments = segment_characters(preprocessed_image)

        if not segments:
            print("No characters were segmented.")
            return 1

        # Concatenate segmented images horizontally for display
        segmented_display = cv2.hconcat(segments)

        # Show the concatenated image
        cv2.imshow("Segmented Characters", segmented_display)
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Close window after key press

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
