import face_recognition
import os
import shutil
import tempfile

dataset_path = r"C:\Users\Lenovo\Desktop\Face_Recognition_Attendance\dataset"

print("[INFO] Checking dataset images for valid faces...\n")

total_images = 0
valid_faces = 0
invalid_faces = 0
temp_dir = tempfile.mkdtemp()

for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder):
        continue

    for file in os.listdir(person_folder):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            total_images += 1
            path = os.path.join(person_folder, file)
            try:
                # Create temp copy to bypass file lock issues
                temp_path = os.path.join(temp_dir, file)
                shutil.copy2(path, temp_path)

                image = face_recognition.load_image_file(temp_path)
                locations = face_recognition.face_locations(image)

                if len(locations) == 1:
                    print(f"✅ {person}/{file}: 1 face detected")
                    valid_faces += 1
                elif len(locations) > 1:
                    print(f"⚠️ {person}/{file}: {len(locations)} faces detected (please keep only one)")
                    invalid_faces += 1
                else:
                    print(f"❌ {person}/{file}: No face detected")
                    invalid_faces += 1
            except Exception as e:
                print(f"[ERROR] Could not process {person}/{file}: {e}")
                invalid_faces += 1

# Cleanup
shutil.rmtree(temp_dir, ignore_errors=True)

print("\n--- SUMMARY ---")
print(f"Total images scanned: {total_images}")
print(f"✅ Valid face images: {valid_faces}")
print(f"❌ Invalid or unreadable images: {invalid_faces}")

if valid_faces == 0:
    print("\n[ALERT] No valid face encodings found! Please ensure each subfolder has clear, single-face images.")
else:
    print("\n[INFO] Dataset is ready to be used for Smart Attendance.")
