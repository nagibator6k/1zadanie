from transformers import pipeline
from PIL import Image
import cv2

# Подставляем модель
detector = pipeline(
    "object-detection",
    model="facebook/detr-resnet-50"  # Твоя модель здесь
)

# Загружаем первый кадр видео
cap = cv2.VideoCapture("4/video.mp4")
ret, frame = cap.read()
cap.release()

image = Image.fromarray(frame)
results = detector(image)

for obj in results:
    print(obj)