import cv2
import numpy as np
import os
import dlib
from deepface import DeepFace

# Database path
db_path = r"C:\Users\hp\Downloads\arpita"

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using dlib
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Ensure bounding box is within frame limits
        x, y = max(0, x), max(0, y)
        w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract face region
        face_roi = frame[y:y+h, x:x+w]

        # Validate ROI size
        if face_roi.size == 0 or face_roi.shape[0] < 48 or face_roi.shape[1] < 48:
            continue

        try:
            # Perform face recognition
            recognition_results = DeepFace.find(
                img_path=face_roi, db_path=db_path,
                model_name="ArcFace", enforce_detection=False
            )

            if recognition_results and len(recognition_results[0]) > 0:
                result = recognition_results[0].iloc[0]  # Get the best match
                identity = os.path.basename(os.path.dirname(result["identity"]))

                # Extract distance and compute confidence (higher distance = lower confidence)
                distance = result["distance"]
                confidence = max(0, 100 - (distance * 100))  # Convert distance to confidence %

            else:
                identity = "unknown"
                confidence = 0

            # Ensure recognized person is displayed as "Arpita"
            if identity.lower() in [folder.lower() for folder in os.listdir(db_path)]:
                identity = "Arpita"

        except Exception as e:
            print(f"Error in face recognition: {e}")
            identity = "unknown"
            confidence = 0

        # Display identity with confidence score
        text = f"{identity} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display method info (adjusted text size)
    cv2.putText(frame, "DeepFace dlib + ArcFace", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
