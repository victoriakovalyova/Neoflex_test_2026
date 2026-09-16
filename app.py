from fastapi import FastAPI
from inference_only import classify

app = FastAPI()

@app.post("/classify")
def classify_endpoint(text: str):
    return classify(text)

# Запуск: uvicorn app:app --host 0.0.0.0 --port 8000
