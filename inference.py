import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === ЗАГРУЗКА ===
MODEL_PATH = "./rubert_classifier"
LABELS = ["technical", "billing", "complaint", "other"]
THRESHOLD = 0.5  # порог уверенности для мультилейбла

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
mlb = joblib.load(f"{MODEL_PATH}/mlb.joblib")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# === ПРЕДСКАЗАНИЕ ===
def classify(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    predicted_labels = [LABELS[i] for i, p in enumerate(probs) if p >= THRESHOLD]
    confidence = {LABELS[i]: float(p) for i, p in enumerate(probs)}
    return {"labels": predicted_labels, "confidence": confidence, "raw_probs": probs}

# Пример
if __name__ == "__main__":
    sample = "У меня сломался личный кабинет, не могу войти уже второй день!"
    result = classify(sample)
    print(result)
