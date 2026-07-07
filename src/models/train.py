import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from config.constants import X_COMBINED_PATH, Y_COMBINED_PATH, MODELS_DIR, XGB_PARAMS, TEST_SPLIT_RATIO

def train_xgboost():
    # Load combined dataset
    if not os.path.exists(X_COMBINED_PATH):
        print("❌ X_combined.npy not found. Run merge_data.py first.")
        return

    X = np.load(X_COMBINED_PATH)
    y = np.load(Y_COMBINED_PATH)

    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"   Classes: {len(np.unique(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_RATIO, random_state=42, stratify=y
    )

    print("🚀 Training XGBoost classifier...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {acc:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'xgb_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Model saved to {model_path}")

    # Also save a symlink or copy as 'model.pkl' for inference scripts expecting that name
    fallback_path = os.path.join(MODELS_DIR, 'model.pkl')
    if not os.path.exists(fallback_path):
        import shutil
        shutil.copyfile(model_path, fallback_path)
        print(f"💾 Also saved as {fallback_path} for compatibility.")

if __name__ == '__main__':
    train_xgboost()