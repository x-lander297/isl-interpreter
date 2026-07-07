import numpy as np
import pickle
from sklearn.dummy import DummyClassifier
from config.constants import X_COMBINED_PATH, Y_COMBINED_PATH, MODELS_DIR

def create_dummy_model():
    X = np.load(X_COMBINED_PATH)
    y = np.load(Y_COMBINED_PATH)

    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'dummy_model.pkl'), 'wb') as f:
        pickle.dump(dummy, f)
    print("✅ Dummy model saved to models/dummy_model.pkl")

if __name__ == '__main__':
    create_dummy_model()