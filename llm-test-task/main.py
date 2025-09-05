import os
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
import time
import csv
import statistics
from typing import List



load_dotenv()

logging.basicConfig(
    filename='server_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="FastAPI LLM Benchmark")

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    logging.error("API ключ OpenRouter не найден.")
    raise ValueError("Необходимо установить OPENROUTER_API_KEY в .env файле")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class GenerationRequest(BaseModel):
    prompt: str
    model: str
    max_tokens: int = 512


class GenerationResponse(BaseModel):
    response: str
    tokens_used: int | None
    latency_seconds: float


def request_with_retry(json_data: dict, max_retries: int = 5, base_delay: float = 1.0):
    """Отправляет POST-запрос с обработкой ошибок 429 (Too Many Requests) и экспоненциальным отсроченным повтором."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    retries = 0
    while retries < max_retries:
        try:
            start_time = time.time()
            response = requests.post(OPENROUTER_URL, headers=headers, json=json_data, timeout=60)
            end_time = time.time()
            latency = round(end_time - start_time, 2)

            if response.status_code == 429:
                delay = base_delay * (2 ** retries)
                logging.warning(f"Получен код 429, повторная попытка через {delay:.2f} с...")
                time.sleep(delay)
                retries += 1
                continue

            response.raise_for_status()

            return response.json(), latency

        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка при обращении к OpenRouter (попытка {retries + 1}/{max_retries}): {e}")
            retries += 1
            if retries < max_retries:
                delay = base_delay * (2 ** retries)
                logging.warning(f"Повторная попытка через {delay:.2f} с...")
                time.sleep(delay)

    logging.error("Не удалось получить успешный ответ после нескольких попыток.")
    raise HTTPException(status_code=500, detail="Не удалось получить успешный ответ от внешнего API.")


@app.get("/models")
def get_models():
    """Возвращает список доступных моделей."""
    return [
        "deepseek/deepseek-chat-v3.1:free",
        "z-ai/glm-4.5-air:free",
        "moonshotai/kimi-k2:free",
    ]


@app.post("/generate", response_model=GenerationResponse)
def generate(request_data: GenerationRequest):
    """Принимает промпт и модель, возвращает ответ от LLM с latency и tokens_used."""
    json_data = {
        "model": request_data.model,
        "messages": [{"role": "user", "content": request_data.prompt}],
        "max_tokens": request_data.max_tokens,
    }

    try:
        result, latency = request_with_retry(json_data)

        model_response = result['choices'][0]['message']['content']
        tokens_used = result.get('usage', {}).get('total_tokens', None)

        return {
            "response": model_response,
            "tokens_used": tokens_used,
            "latency_seconds": latency
        }
    except (KeyError, IndexError) as e:
        logging.error(f"Неверный формат ответа от OpenRouter: {result}. Ошибка: {e}")
        raise HTTPException(status_code=500, detail="Неверный формат ответа от внешнего API.")


@app.post("/benchmark")
async def benchmark(
        model: str = Form(...),
        runs: int = Form(5),
        prompt_file: UploadFile = File(...)
):
    """Проводит бенчмарк модели на основе файла с промптами."""
    contents = await prompt_file.read()
    try:
        prompts = [line for line in contents.decode('utf-8').strip().split('\n') if line.strip()]
    except Exception as e:
        logging.error(f"Ошибка при чтении файла: {e}")
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать файл: {e}")

    if not prompts:
        raise HTTPException(status_code=400, detail="Файл с промптами пуст или не содержит корректных строк.")

    all_latencies = []

    for prompt in prompts:
        for i in range(runs):
            logging.info(f"Выполняется бенчмарк для промпта '{prompt[:30]}...' (попытка {i + 1}/{runs})")
            json_data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
            }
            try:
                _, latency = request_with_retry(json_data)
                all_latencies.append(latency)
            except HTTPException:
                logging.warning("Пропуск итерации бенчмарка из-за ошибки запроса.")
                continue

    if not all_latencies:
        raise HTTPException(status_code=500,
                            detail="Не удалось выполнить ни одного успешного запроса в бенчмарке. Проверьте логи.")

    stats = {
        "model": model,
        "total_runs": len(all_latencies),
        "avg_latency": round(statistics.mean(all_latencies), 2),
        "min_latency": round(min(all_latencies), 2),
        "max_latency": round(max(all_latencies), 2),
        "std_dev_latency": round(statistics.stdev(all_latencies), 2) if len(all_latencies) > 1 else 0.0
    }

    with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        writer.writeheader()
        writer.writerow(stats)

    logging.info(f"Бенчмарк для модели '{model}' завершен. Результаты сохранены в benchmark_results.csv")
    return stats


@app.get("/")
def root():
    return {
        "message": "FastAPI LLM Test Server",
        "instructions": {
            "models_endpoint": "GET /models — получить список доступных моделей",
            "generate_endpoint": "POST /generate — отправить prompt и модель, чтобы получить ответ LLM",
            "benchmark_endpoint": "POST /benchmark — провести бенчмарк моделей",
            "docs": "Открой /docs для интерактивного тестирования эндпоинтов"
        }
    }
