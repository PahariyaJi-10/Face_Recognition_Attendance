import face_recognition
import os

# Path to your dataset folder
dataset_path = r"C:\Users\Lenovo\Desktop\Face_Recognition_Attendance\dataset"

# Check if folder exists
if not os.path.exists(dataset_path):
    print(f"[ERROR] Dataset folder not found at: {dataset_path}")
    exit()

print("[INFO] Checking dataset images for valid faces...\n")

# Counters
total_images = 0
valid_faces = 0

# Loop through all image files
for file in os.listdir(dataset_path):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        total_images += 1
        path = os.path.join(dataset_path, file)
        try:
            # Load image and find faces
            image = face_recognition.load_image_file(path)
            locations = face_recognition.face_locations(image)

            if len(locations) > 0:
                valid_faces += 1
                print(f"[OK] {file}: {len(locations)} face(s) detected ✅")
            else:
                print(f"[WARN] {file}: No faces detected ⚠️")

        except Exception as e:
            print(f"[ERROR] Could not process {file}: {e}")

# Summary
print("\n--- SUMMARY ---")
print(f"Total images scanned: {total_images}")
print(f"Images with valid faces: {valid_faces}")
print(f"Images without faces: {total_images - valid_faces}")

if valid_faces == 0:
    print("\n[ALERT] No valid face encodings found! Please add clear single-face images to the dataset folder.")


