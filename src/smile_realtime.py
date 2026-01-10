import cv2
import numpy as np
import tensorflow as tf
from smile_model import build_smile_model

# ✅ Load trained model
model = build_smile_model()
model.load_weights("../models/smile_model.h5")   # make sure path is correct

# ✅ Class labels
class_labels = ["No Smile", "Smile"]

# ✅ Initialize webcam
cap = cv2.VideoCapture(0)

# ✅ Haar Cascade (for face detection only, NOT smile detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))  # ✅ match model input size
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)   # add channel dim
        face = np.expand_dims(face, axis=0)    # add batch dim

        # ✅ Prediction
        preds = model.predict(face, verbose=0)[0]
        label_index = np.argmax(preds)
        confidence = np.max(preds)

        # ✅ Direct prediction (no threshold filtering)
        label = class_labels[label_index]

        # ✅ Draw bounding box + label
        color = (0, 255, 0) if label == "Smile" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

    cv2.imshow("Smile Detection (CNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()