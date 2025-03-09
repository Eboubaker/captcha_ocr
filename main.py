from fileinput import filename
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from lib import preprocess_image, segment_characters, CAPTCHA_LENGTH, predict_captcha, CHARACTERS, CHAR_CANVAS_SIZE

# Define constants
IMG_HEIGHT = CHAR_CANVAS_SIZE
IMG_WIDTH = CHAR_CANVAS_SIZE
BATCH_SIZE = 32
EPOCHS = 30

def load_dataset(data_dirs):
    """Load and preprocess the dataset."""
    images = []
    labels = []
    for image_path in get_images(data_dirs):
        label = get_label(image_path)
        img = preprocess_image(image_path)
        segments = segment_characters(img)
        if len(segments) != CAPTCHA_LENGTH:
            # print(f"{img_path}: WARN: I only found {len(segments)}")
            continue
        for i, seg in enumerate(segments):
            images.append(seg)
            labels.append(CHARACTERS.find(label[i]))        
    return np.array(images), np.array(labels)

def get_images(data_dirs):
    for data_dir in data_dirs:
        for filename in os.listdir(data_dir):
            if filename.endswith('.png'):
                yield os.path.join(data_dir, filename)     

                
def get_label(image_path):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    return filename

def load_dataset_unsegmented(data_dirs):
    """Load and preprocess the dataset."""
    images = []
    labels = []
    for image_path in get_images(data_dirs):
        img = preprocess_image(image_path)
        label = get_label(image_path)
        images.append(img)
        labels.append(label) 
    return np.array(images), np.array(labels)

def build_model():
    """Build a CNN model for CAPTCHA recognition."""
    # Input layer
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer - one output for each character position
    #outputs = []
    #for i in range(MAX_LENGTH):
    output = layers.Dense(len(CHARACTERS), activation='softmax')(x)
    #outputs.append(out)
    
    # Create and compile model
    model = models.Model(inputs=inputs, outputs=output)
        
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model."""
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")

    # Reshape images for the model
    X_train = X_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    X_val = X_val.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    
    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
        
    # Create a data augmentation function
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    # Train using fit method with dictionaries
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr]
    )
    
    return history


def calc_accuracy(model, images):
    """Evaluate the model on test data."""
    total = len(images)
    correct = 0

    def process_image(image):
        label = get_label(image)
        predicted = predict_captcha(model, image)
        if label.lower() != predicted.lower():
            print("failed on ", image)
        return 1 if label.lower() == predicted.lower() else 0

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, images), total=total, desc="Processing"))

    correct = sum(results)
    return correct / total if total > 0 else 0
    
# Main execution
if __name__ == "__main__":
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    images, labels = load_dataset(['correct', 'labeled'])  # all captchas images
    print(f"Loaded {len(images)} images and {len(labels)} labels")
    
    # Check for any mismatched data
    if len(images) != len(labels):
        print("WARNING: Number of images and labels don't match!")
    
    # 2. Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 4. Build model
    print("Building model...")
    model = build_model()
    model.summary()
    
    # 5. Train model
    print("Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # 6. Evaluate model
    # print("Evaluating model...")
    overall_accuracy = 100 * calc_accuracy(model, list(get_images(['correct', 'labeled'])))
    print(f"Model accuracy: {overall_accuracy:.2f}%")
    
    # 7. Save model
    model.save("captcha_model.keras", include_optimizer=True)
    print("Model saved as 'captcha_model.keras'")

    # model.save("captcha_model.h5", include_optimizer=True)
    # print("Model saved as 'captcha_model.h5'")
    # model.export("captcha_model")
    # print("Model saved as 'captcha_model'")

    # 8. Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot accuracy for the first character position
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()