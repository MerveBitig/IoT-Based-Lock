import cv2
import mediapipe as mp
# Eskisi yerine bu direkt yolları kullanalım:
from mediapipe.python.solutions import face_detection as mp_face_detection
from mediapipe.python.solutions import drawing_utils as mp_drawing

# 2. Kamerayı Başlat
cap = cv2.VideoCapture(0)

print("Akilli Kilit Sistemi: Goruntu Isleme Aktif...")

# Buradaki face_detection kullanımını da mp_face_detection olarak güncelledik:
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        cv2.imshow('IoT Akilli Kilit', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
