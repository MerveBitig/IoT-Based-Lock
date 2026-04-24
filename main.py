import cv2
import mediapipe as mp

# En standart çağırma şekli
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    print("Kamera aciliyor, lutfen bekleyin...")
    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
        
        cv2.imshow('IoT Kilit Sistemi', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
