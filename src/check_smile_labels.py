import numpy as np

X_smile = np.load("C:\\Infosys springboard\\preprocessed\\X_smile.npy")
y_smile = np.load("C:\\Infosys springboard\\preprocessed\\y_smile.npy")

print("Unique labels in smile dataset:", set(y_smile))
print("Count of smile (1):", (y_smile == 1).sum())
print("Count of non-smile (0):", (y_smile == 0).sum())