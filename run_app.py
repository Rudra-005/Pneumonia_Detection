#!/usr/bin/env python3
"""
Startup script for Pneumonia Detection Application
"""
import subprocess
import sys
import os

def main():
    """Main function to start the application"""
    print("PNEUMONIA DETECTION APPLICATION")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("UI/Home_Page.py"):
        print("UI/Home_Page.py not found. Please run from project root.")
        return
    
    # Check if models exist
    if not os.path.exists("Saved_Models"):
        print("Saved_Models directory not found. Please run training first:")
        print("   python quick_train.py")
        return
    
    print("Starting Streamlit application...")
    print("The app will open in your browser at: http://localhost:8502")
    print("To stop the app, press Ctrl+C")
    print("-" * 50)
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "UI/Home_Page.py", 
            "--server.port", "8502",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    main()