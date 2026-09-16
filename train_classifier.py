import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset

# === КОНФИГ ===
MODEL_NAME = "cointegrated/rubert-base-cased"  # или "blanchefort/rubert-base-cased-sentiment"
MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
LABELS = ["technical", "billing", "complaint", "other"]  # подставь свои классы

# === ЗАГРУЗКА ДАННЫХ ===
# Ожидаем CSV с колонками: text, labels (через запятую, например: "technical,billing")
df = pd.read_csv("support_tickets.csv")

# === ПРЕДОБРАБОТКА ===
mlb = MultiLabelBinarizer(classes=LABELS)
y = mlb.fit_transform(df["labels"].str.split(",").apply(lambda x: [l.strip() for l in x]))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].fillna("").tolist(), y, test_size=0.2, random_state=42, stratify=y.sum(axis=1)
)

# === ТОКЕНИЗАЦИЯ ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts, labels):
    enc = tokenizer(texts, truncation=True, padding=False, max_length=MAX_LENGTH)
    return {**enc, "labels": labels}

train_dataset = Dataset.from_dict(tokenize(train_texts, train_labels.tolist()))
val_dataset = Dataset.from_dict(tokenize(val_texts, val_labels.tolist()))

# === МОДЕЛЬ ===
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="multi_label_classification"
)

# === ТРЕНИРОВКА ===
training_args = TrainingArguments(
    output_dir="./rubert_classifier",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

trainer.train()
trainer.save_model("./rubert_classifier")
mlb.save("./rubert_classifier/mlb.joblib")  # сохраним бинаризатор меток
