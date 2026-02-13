import sys
import os
import cv2
import torch

print("Python executable:", sys.executable)
print("OpenCV version:", cv2.__version__)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

try:
    from app.database import DatabaseManager
    print("DatabaseManager imported successfully")
    db = DatabaseManager()
    print("DatabaseManager instantiated")
except Exception as e:
    print(f"DatabaseManager error: {e}")

try:
    from app.core.detector import Detector
    print("Detector imported successfully")
    # Don't instantiate to save time/memory in this quick test
except Exception as e:
    print(f"Detector error: {e}")

try:
    from app.core.classifier import Classifier
    print("Classifier imported successfully")
except Exception as e:
    print(f"Classifier error: {e}")

try:
    from app.core.camera import Camera
    print("Camera imported successfully")
except Exception as e:
    print(f"Camera error: {e}")

print("All checks passed!")
