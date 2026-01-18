#!/usr/bin/env python3
"""
Minimal training script for pneumonia detection models
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50, MobileNetV2, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import cv2
from PIL import Image
import pydicom

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5  # Minimal for quick training
NUM_CLASSES = 2

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    
    # Load CSV with labels
    csv_path = "Data/stage_2_detailed_class_info.csv"
    df = pd.read_csv(csv_path)
    
    # Simplify labels: Normal vs Pneumonia
    df['binary_label'] = df['class'].apply(lambda x: 0 if x == 'Normal' else 1)
    
    # Get image paths and labels
    image_paths = []
    labels = []
    
    # Check Images directory for available files
    images_dir = "Images"
    available_files = []
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.endswith(('.dcm', '.jpg', '.png', '.jpeg')):
                available_files.append(file)
    
    print(f"Found {len(available_files)} image files")
    
    # Match CSV entries with available files
    for _, row in df.iterrows():
        patient_id = row['patientId']
        # Look for matching files
        matching_files = [f for f in available_files if patient_id in f]
        if matching_files:
            image_paths.append(os.path.join(images_dir, matching_files[0]))
            labels.append(row['binary_label'])
    
    print(f"Matched {len(image_paths)} images with labels")
    return image_paths[:100], labels[:100]  # Limit for quick training

def load_image(image_path):
    """Load and preprocess a single image"""
    try:
        if image_path.endswith('.dcm'):
            # Load DICOM
            dcm = pydicom.dcmread(image_path, force=True)
            img_array = dcm.pixel_array.astype(np.float32)
            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            img = Image.fromarray(img_array).convert('RGB')
        else:
            # Load regular image
            img = Image.open(image_path).convert('RGB')
        
        # Resize and normalize
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE, 3))

def create_basic_cnn():
    """Create basic CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_transfer_model(base_model_name):
    """Create transfer learning model"""
    if base_model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif base_model_name == 'mobilenetv2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif base_model_name == 'densenet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze base model
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_name):
    """Train a model"""
    print(f"\nTraining {model_name}...")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Save model
    model_path = f"Saved_Models/{model_name}.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, history

def main():
    """Main training function"""
    print("Starting pneumonia detection model training...")
    
    # Load data
    image_paths, labels = load_and_preprocess_data()
    
    if len(image_paths) == 0:
        print("No images found! Please check the Images directory.")
        return
    
    # Load images
    print("Loading and preprocessing images...")
    X = []
    y = []
    
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_paths)}")
        
        img = load_image(path)
        if img is not None:
            X.append(img)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} images")
    print(f"Normal: {np.sum(y == 0)}, Pneumonia: {np.sum(y == 1)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    models_to_train = [
        ('basic_cnn_model', create_basic_cnn()),
        ('resnet50_model', create_transfer_model('resnet50')),
        ('transfer_learning_model_mobilenetv2', create_transfer_model('mobilenetv2')),
        ('transfer_learning_model_densenet121', create_transfer_model('densenet121'))
    ]
    
    for model_name, model in models_to_train:
        try:
            trained_model, history = train_model(model, X_train, y_train, X_val, y_val, model_name)
            
            # Evaluate
            val_loss, val_acc = trained_model.evaluate(X_val, y_val, verbose=0)
            print(f"{model_name} - Validation Accuracy: {val_acc:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    print("\nTraining completed!")
    print("Models saved in Saved_Models/ directory")

if __name__ == "__main__":
    main()