import cv2
import numpy as np
import os

# Ensure the data directory exists
dataset_path = './data/'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    print(f"Created directory: {dataset_path}")

# Initialize camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

skip = 0
face_data = []
file_name = input("Enter the name of the person: ")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)  # Reverse sorting gives the largest face first

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extract (crop out the required face): region of interest
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        skip += 1
        if skip % 10 == 0:
            face_data.append(face_section)
            print(f"Captured {len(face_data)} face images")

    cv2.imshow("Frame", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q') or len(face_data) >= 50:  # Limit to 50 images
        break

# Convert face list to a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(f"Face data shape: {face_data.shape}")

# Save this data into the file system
file_path = os.path.join(dataset_path, file_name + '.npy')
np.save(file_path, face_data)
print(f"Data successfully saved at {file_path}")

cap.release()
cv2.destroyAllWindows()
