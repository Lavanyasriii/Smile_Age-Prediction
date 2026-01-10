import cv2
import os
import numpy as np

# Paths
smile_path = "C:\\Infosys springboard\\data\\smile\\kaggle-genki4k"
age_path = "C:\\Infosys springboard\\data\\age\\UTKFace"

# Output folder
output_path = "C:\\Infosys springboard\\preprocessed"
os.makedirs(output_path, exist_ok=True)

# --- Helper to preprocess image ---
def preprocess_image(img, size=(64, 64)):
    img = cv2.resize(img, size)           # resize to fixed size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
    img = img.astype("float32") / 255.0   # normalize 0–1
    return img

# --- Smile Dataset ---
X_smile = []
y_smile = []

for label, folder in enumerate(["non_smile", "smile"]):  # 0 = non-smile, 1 = smile
    folder_path = os.path.join(smile_path, folder)
    for f in os.listdir(folder_path):
        img_path = os.path.join(folder_path, f)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping corrupted {f}")
            continue
        img = preprocess_image(img)
        X_smile.append(img)
        y_smile.append(label)

X_smile = np.array(X_smile).reshape(-1, 64, 64, 1)  # shape: (N, 64, 64, 1)
y_smile = np.array(y_smile)

np.save(os.path.join(output_path, "X_smile.npy"), X_smile)
np.save(os.path.join(output_path, "y_smile.npy"), y_smile)

print(f"Smile dataset saved: {X_smile.shape}, Labels: {y_smile.shape}")


# --- Age Dataset ---
X_age = []
y_age = []

for f in os.listdir(age_path):
    img_path = os.path.join(age_path, f)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping corrupted {f}")
        continue
    try:
        age = int(f.split("_")[0])  # filename format: age_gender_race_date.jpg
    except:
        print(f"Skipping invalid filename: {f}")
        continue

    img = preprocess_image(img)
    X_age.append(img)
    y_age.append(age)

X_age = np.array(X_age).reshape(-1, 64, 64, 1)
y_age = np.array(y_age)

np.save(os.path.join(output_path, "X_age.npy"), X_age)
np.save(os.path.join(output_path, "y_age.npy"), y_age)

print(f"Age dataset saved: {X_age.shape}, Labels: {y_age.shape}")