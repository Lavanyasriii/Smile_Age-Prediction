import numpy as np
from tensorflow.keras.utils import to_categorical
import os

# Path to splits folder
split_path = "C:\\Infosys springboard\\splits"

# --- Load Smile Dataset ---
X_smile_train = np.load(os.path.join(split_path, "X_smile_train.npy"))
y_smile_train = np.load(os.path.join(split_path, "y_smile_train.npy"))
X_smile_val = np.load(os.path.join(split_path, "X_smile_val.npy"))
y_smile_val = np.load(os.path.join(split_path, "y_smile_val.npy"))
X_smile_test = np.load(os.path.join(split_path, "X_smile_test.npy"))
y_smile_test = np.load(os.path.join(split_path, "y_smile_test.npy"))

# One-hot encode labels (0 = non-smile, 1 = smile)
y_smile_train = to_categorical(y_smile_train, num_classes=2)
y_smile_val = to_categorical(y_smile_val, num_classes=2)
y_smile_test = to_categorical(y_smile_test, num_classes=2)

print("Smile dataset:")
print("Train:", X_smile_train.shape, y_smile_train.shape)
print("Val:", X_smile_val.shape, y_smile_val.shape)
print("Test:", X_smile_test.shape, y_smile_test.shape)

# --- Load Age Dataset ---
X_age_train = np.load(os.path.join(split_path, "X_age_train.npy"))
y_age_train = np.load(os.path.join(split_path, "y_age_train.npy"))
X_age_val = np.load(os.path.join(split_path, "X_age_val.npy"))
y_age_val = np.load(os.path.join(split_path, "y_age_val.npy"))
X_age_test = np.load(os.path.join(split_path, "X_age_test.npy"))
y_age_test = np.load(os.path.join(split_path, "y_age_test.npy"))

print("\nAge dataset:")
print("Train:", X_age_train.shape, y_age_train.shape)
print("Val:", X_age_val.shape, y_age_val.shape)
print("Test:", X_age_test.shape, y_age_test.shape)
