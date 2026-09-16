import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# === КОНФИГ ===
MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"  # предобученная русская модель
LABELS = ["technical", "billing", "complaint", "other"]
THRESHOLD = 0.5

# === ЗАГРУЗКА МОДЕЛИ ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# === КЛАССИФИКАЦИЯ ===
def classify(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # Для мультилейбла используем порог, для сингл-лейбла — argmax
    predicted_idx = np.argmax(probs)
    predicted_label = LABELS[predicted_idx]
    confidence = {LABELS[i]: float(p) for i, p in enumerate(probs)}
    
    return {"label": predicted_label, "confidence": confidence, "raw_probs": probs}

# === ПРИМЕР ===
if __name__ == "__main__":
    samples = [
        "У меня сломался личный кабинет, не могу войти!",
        "Почему списали 500 руб за подписку?",
        "Ужасный сервис, уже 7 дней жду ответа!",
        "Спасибо за помощь, всё работает!",
    ]
    for text in samples:
        result = classify(text)
        print(f"Текст: {text}")
        print(f"Предсказание: {result['label']} (уверенность: {result['confidence'][result['label']]:.2%})\n")
