import cv2
import numpy as np
import os
from deepface import DeepFace

# Database path
db_path = r"C:\Users\hp\Downloads\arpita"

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using DeepFace's built-in MTCNN
    face_objs = DeepFace.extract_faces(
        img_path=frame, detector_backend="mtcnn", enforce_detection=False
    )

    for face_obj in face_objs:
        facial_area = face_obj["facial_area"]
        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

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
    cv2.putText(frame, "DeepFace mtcnn + ArcFace", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
