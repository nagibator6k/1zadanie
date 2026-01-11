from transformers import pipeline

classifier = pipeline("sentiment-analysis",
                      model='tabularisai/multilingual-sentiment-analysis')

text = "Kirieshki are tasty"
result = classifier(text)
print("Текст:", text)
print("Результат:", result)