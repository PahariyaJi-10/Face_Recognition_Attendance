# 🧑‍💻 Face Recognition Attendance System

This project is an **AI-based Smart Attendance System** that uses **face recognition** to automatically mark attendance of registered students or employees. It eliminates the need for manual entry, enhances accuracy, and ensures a contactless, efficient process.

---

## 🚀 Features

- 🔍 **Face Detection & Recognition** using OpenCV and Haar Cascade / LBPH algorithms  
- 🧑‍🏫 **Automatic Attendance Marking** once a known face is detected  
- 💾 **Dataset Creation Module** to register new users (face capture and labeling)  
- 📅 **Date & Time-based Attendance Storage** (CSV or Excel format)  
- 🔒 **Liveness Detection** (optional) to prevent spoofing using photos  
- 💻 **Simple GUI** for user interaction (optional Tkinter / Streamlit / Flask UI)  
- 📸 **Real-time Camera Integration**

---

## 🧩 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Data Handling | NumPy, Pandas |
| Storage | CSV / Excel / Database |
| Optional | Tkinter, Flask, Streamlit |
| Environment | Jupyter Notebook / VS Code / PyCharm |

---

## ⚙️ How It Works

1. **Dataset Creation**  
   - Capture multiple face images per person using webcam  
   - Store them in a dataset folder with user ID and name  

2. **Training Phase**  
   - Train a facial recognition model (e.g., LBPHFaceRecognizer) using the dataset  

3. **Recognition & Attendance**  
   - Detect faces in real time  
   - Match with trained model  
   - If recognized, mark attendance with name, ID, date, and time  

4. **View Records**  
   - Attendance stored in `.csv` file (e.g., `Attendance_2025-11-13.csv`)  
   - Can be viewed in Excel or a GUI dashboard
  



🌟 Future Enhancements

🔐 Cloud-based database integration (Firebase/MySQL)

📱 Android app version using API

🎯 Improved face recognition with deep learning (FaceNet / Dlib)

👁️ Real-time liveness and mask detection

📈 Web dashboard for viewing attendance reports
