FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и датасет
COPY . .

# Генерируем датасет если нет
RUN if [ ! -f support_tickets_synthetic.csv ]; then \
    python generate_dataset.py; \
    fi

# Обучаем модель при сборке (опционально, можно убрать для CI/CD)
RUN if [ -f train_classifier.py ]; then \
    python train_classifier.py; \
    fi

# Порт для API
EXPOSE 8000

# Запуск API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
