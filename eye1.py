import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
import datetime

# Initialize sound alert
pygame.mixer.init()
pygame.mixer.music.load("South.mp3")  # Ensure "alert.mp3" is in the working directory

# Load Dlib's face detector and 68-landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Parameters
EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 20
frame_count = 0
alert_active = False
total_frames = 0
drowsy_frames = 0

# Open log file
log_file = open("drowsiness_log.txt", "a")

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Draw eye landmarks
        for (x, y) in np.vstack((left_eye, right_eye)):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        if avg_EAR < EAR_THRESHOLD:
            frame_count += 1
            drowsy_frames += 1
            if frame_count >= CLOSED_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                print("DROWSINESS ALERT! Wake up!")

                # Calculate drowsiness percentage
                drowsiness_percentage = (drowsy_frames / total_frames) * 100 if total_frames > 0 else 0
                log_file.write(f"Drowsiness detected at {datetime.datetime.now()} | Drowsiness %: {drowsiness_percentage:.2f}%\n")
                log_file.flush()

                if not alert_active:
                    pygame.mixer.music.play(-1)
                    alert_active = True
        else:
            frame_count = 0
            if alert_active:
                pygame.mixer.music.stop()
                alert_active = False

    # Calculate and display drowsiness percentage
    drowsiness_percentage = (drowsy_frames / total_frames) * 100 if total_frames > 0 else 0
    percentage_text = f"Drowsiness: {drowsiness_percentage:.2f}%"
    cv2.putText(frame, percentage_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Display frame
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
