from transformers import pipeline
from PIL import Image
image_classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"
)
image_path = "C:/Users/vanak/OneDrive/Documenti/GitHub/1zadanie/3/kruzhka.jpg"
image = Image.open(image_path).convert("RGB")
result = image_classifier(image)
print(result)
