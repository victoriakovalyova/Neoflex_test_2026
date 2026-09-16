AI-классификатор обращений поддержки
Классификация тикетов (technical / billing / complaint / other) на базе RuBERT с API для интеграции в службу поддержки.

Возможности
Мультилейбл-классификация текстов обращений

Синтетический датасет для быстрого старта

Обучение и инференс на CPU/GPU

FastAPI-эндпоинт для интеграции

Задержка <2 с при правильной оптимизации

Быстрый старт
bash
# Сборка и запуск
docker build -t support-classifier .
docker run -p 8000:8000 support-classifier

# Тест API
curl -X POST "http://localhost:8000/classify?text=У%20меня%20сломался%20личный%20кабинет"
Структура
generate_dataset.py — генерация синтетических тикетов

train_classifier.py — обучение RuBERT

inference.py — предсказание

app.py — FastAPI-сервер

Dockerfile, requirements.txt — развёртывание

Данные
По умолчанию генерируется support_tickets_synthetic.csv (10k примеров). Для продакшена замените на реальные размеченные тикеты.

API
POST /classify?text=<сообщение>
Ответ: {labels: [...], confidence: {...}, raw_probs: [...]}

Требования
Python 3.11+

Docker (опционально)

