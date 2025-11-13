Python 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import cv2
... import os
... import numpy as np
... 
... # Path to dataset folder
... dataset_path = "dataset"
... 
... faces = []
... labels = []
... label_map = {}   # maps ID number → student name
... current_id = 0
... 
... # Haar Cascade for face detection
... face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
... 
... # Loop through each student's folder
... for student_name in os.listdir(dataset_path):
...     student_folder = os.path.join(dataset_path, student_name)
...     if not os.path.isdir(student_folder):
...         continue
... 
...     # Map numeric ID to student name
...     label_map[current_id] = student_name
... 
...     # Loop through each image in the student folder
...     for image_name in os.listdir(student_folder):
...         image_path = os.path.join(student_folder, image_name)
...         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
... 
...         if img is None:
...             continue
... 
...         faces.append(img)
...         labels.append(current_id)
... 
...     current_id += 1
... 
... # Convert lists to numpy arrays
... faces = np.array(faces, dtype="object")
... labels = np.array(labels)
... 
... # Create and train the LBPH face recognizer
... recognizer = cv2.face.LBPHFaceRecognizer_create()
... recognizer.train(faces, labels)
... 
# Save the trained model and label map
recognizer.write("face_model.yml")
np.save("labels.npy", label_map)

print("✅ Training complete!")
print("Model saved as face_model.yml")
print("Labels saved as labels.npy")
