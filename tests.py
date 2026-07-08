# test_camera_quick.py
import logging
import sys
from src.inference.camera import Camera

logging.basicConfig(level=logging.INFO)

def main():
    print("📷 Testing Camera Module...")
    
    cam = Camera(camera_id=0, width=640, height=480)
    
    if not cam.start():
        print("❌ Camera failed to start. Try camera_id=1")
        return False
    
    print("✅ Camera started")
    
    # Try to read 5 frames
    for i in range(5):
        frame = cam.read()
        if frame is not None:
            print(f"✅ Frame {i+1}: Shape={frame.shape}, dtype={frame.dtype}")
        else:
            print(f"⚠️ Frame {i+1}: None (skipped or failed)")
    
    cam.release()
    print("✅ Camera released")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)