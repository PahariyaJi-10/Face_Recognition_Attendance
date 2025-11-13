import face_recognition
import cv2
import os
import numpy as np
import pickle

print("[INFO] Loading images and encoding faces...")

dataset_path = r"C:\Users\Lenovo\Desktop\Face_Recognition_Attendance\dataset"
known_encodings = []
known_names = []

# Loop through all folders (each folder = one person)
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Processing {person_name}...")
    for file in os.listdir(person_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(person_folder, file)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
            else:
                print(f"[WARNING] No face found in {file}")

# Save encodings
data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Encoding complete! Total known faces: {len(known_names)}")
