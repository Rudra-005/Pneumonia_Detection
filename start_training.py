#!/usr/bin/env python3
"""
Simple script to start model training
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def train_cnn_models():
    """Train CNN models"""
    print("\n" + "="*50)
    print("TRAINING CNN MODELS")
    print("="*50)
    
    try:
        subprocess.check_call([sys.executable, "train_models.py"])
        print("CNN models training completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error training CNN models: {e}")
        return False

def setup_yolo():
    """Setup YOLOv5"""
    print("\n" + "="*50)
    print("SETTING UP YOLOV5")
    print("="*50)
    
    try:
        subprocess.check_call([sys.executable, "train_yolo.py"])
        print("YOLOv5 setup completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up YOLOv5: {e}")
        return False

def main():
    """Main function"""
    print("PNEUMONIA DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("requirements.txt not found. Please run from project root directory.")
        return
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Exiting.")
        return
    
    # Train CNN models
    cnn_success = train_cnn_models()
    
    # Setup YOLOv5
    yolo_success = setup_yolo()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"CNN Models: {'Success' if cnn_success else 'Failed'}")
    print(f"YOLOv5 Setup: {'Success' if yolo_success else 'Failed'}")
    
    if cnn_success or yolo_success:
        print("\nTraining completed! You can now run the application:")
        print("   streamlit run UI/Home_Page.py")
    else:
        print("\nTraining failed. Please check the errors above.")

if __name__ == "__main__":
    main()