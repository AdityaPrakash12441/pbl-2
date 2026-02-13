"""
Test script to check camera availability
"""
import cv2
import sys

print("Testing camera access...")
print("-" * 50)

# Try to open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    print("\nPossible issues:")
    print("1. Camera is being used by another application")
    print("2. No camera permissions granted")
    print("3. No camera detected")
    print("\nTo fix:")
    print("- Close other apps using camera (Zoom, FaceTime, etc.)")
    print("- Go to: System Settings > Privacy & Security > Camera")
    print("- Enable camera access for Terminal/iTerm")
    sys.exit(1)

# Try to read a frame
ret, frame = cap.read()

if not ret:
    print("❌ Camera opened but cannot read frames")
    print("\nTry:")
    print("- Restart your terminal")
    print("- Check camera permissions")
    cap.release()
    sys.exit(1)

# Success
print("✅ Camera is accessible!")
print(f"✅ Frame size: {frame.shape[1]}x{frame.shape[0]}")
print(f"✅ Camera backend: {cap.getBackendName()}")
print("\nYou can now run: ./run.sh")

cap.release()
