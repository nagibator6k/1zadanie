from transformers import pipeline
from PIL import Image

# Создаём классификатор
image_classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"  # подставь свою модель
)

# Загружаем изображение через PIL
image_path = "C:/Users/vanak/OneDrive/Documenti/GitHub/1zadanie/3/kruzhka.jpg"
image = Image.open(image_path).convert("RGB")  # обязательно RGB

# Передаём объект PIL в pipeline
result = image_classifier(image)
print(result)
