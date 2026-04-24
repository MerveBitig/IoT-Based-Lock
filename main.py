import cv2
import mediapipe as mp

# En garanti yol: Doğrudan sınıfı içe aktaralım
FaceDetection = mp.solutions.face_detection
DrawingUtils = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Kamera kontrolü
if not cap.isOpened():
    print("HATA: Kamera bulunamadi veya baska bir uygulama kullaniyor!")
    exit()

print("Sistem calisiyor... Kapatmak icin kamera ekranindayken 'q' tusuna basin.")

with FaceDetection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Görüntü işleme
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                DrawingUtils.draw_detection(image, detection)
                print("Yuz algilandi!")

        cv2.imshow('Akilli Kilit Yuz Tespit', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
