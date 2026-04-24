import cv2
import mediapipe as mp

# mp.solutions yerine doğrudan alt paketlerden çağırıyoruz:
from mediapipe.python.solutions import face_detection as mp_face_detection
from mediapipe.python.solutions import drawing_utils as mp_drawing

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("HATA: Kamera acilamadi!")
    exit()

print("Sistem aktif... Kapatmak icin 'q' tusuna basin.")

# Burada da yeni ismi kullanıyoruz:
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                print("Yuz algilandi!")

        cv2.imshow('Akilli Kilit', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
