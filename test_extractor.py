import sys
from src.data.loader import load_static_images
from src.features.extractor import get_landmarks_from_image

def main():
    print("📂 Loading images...")
    X, y = load_static_images()
    print(f"✅ Loaded {len(X)} images.")

    print("🔍 Testing extractor on first 20 images...")
    success_count = 0
    for i in range(min(20, len(X))):
        feat = get_landmarks_from_image(X[i])
        if feat is not None:
            print(f"✅ Image {i}: Feature shape {feat.shape}")
            success_count += 1
        else:
            print(f"❌ Image {i}: No hand detected")

    print(f"\n📊 Extractor succeeded on {success_count}/{min(20, len(X))} images.")
    if success_count > 0:
        print("✅ extractor.py passed! Proceed to File #4.")
        sys.exit(0)
    else:
        print("⚠️ No hand detected in first 20 images. This might be normal if the dataset contains images without hands. Try increasing the range to 100 images.")
        sys.exit(1)

if __name__ == "__main__":
    main()