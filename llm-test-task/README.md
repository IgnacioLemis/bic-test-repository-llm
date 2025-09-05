# Тестовое задание: FastAPI + LLM Benchmark

## Уровни выполнения

### Уровень 1 — Минимально рабочее решение

* [ ] Поднять FastAPI-сервер.
* [ ] `GET /models` — вернуть список моделей, например:

  ```
  deepseek/deepseek-chat-v3.1:free
  z-ai/glm-4.5-air:free
  moonshotai/kimi-k2:free
  ```
* [ ] `POST /generate`:

  * принимает `prompt`, `model`;
  * вызывает OpenRouter (OpenAI-совместимый формат) через `requests.post`;
  * возвращает ответ модели (текст).
* [ ] API-ключ: из `.env` (через `dotenv`)
* [ ] Логирование ошибок в `server_logs.txt`.

---

### Уровень 2 — Продвинутая реализация

* [ ] `POST /generate`:

  * параметр `max_tokens` (по умолчанию 512);
  * вернуть JSON с `response`, `tokens_used`, `latency_seconds`;
  * глобальный обработчик 429 с экспоненциальным `retry`.
* [ ] `POST /benchmark`:

  * вход: `prompt_file` (txt, по строкам), `model`, `runs` (int, дефолт 5);

  * метрики latency: `avg`, `min`, `max`, `std_dev`;
  * сохранить `benchmark_results.csv`;
  * вернуть JSON со статистикой.

---

### Уровень 3 — Профессиональное решение

* [ ] `/generate`: при `stream=true` — **SSE-стриминг**.
* [ ] `/benchmark`: при `visualize=true` — вернуть **HTML-таблицу**.
* [ ] Провести 10 тестов: 5 для `/generate` (разные модели, включая stream), 5 для `/benchmark` (≥3 промпта, разные `runs`).
* [ ] Замерить общую latency (например, `curl -w "%{time_total}"`).
* [ ] Сравнительная таблица моделей (средняя задержка + `std_dev`).
* [ ] Приложить: `benchmark_results.csv`, `server_logs.txt`, скриншоты ответов.

---

## Примеры cURL (для проверки)

**1) Получить список моделей**

```bash
curl -s "http://127.0.0.1:8000/models"
```

**2) Генерация (базовый POST JSON)**

```bash
curl -sS -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Напиши стих про Python","model":"deepseek/deepseek-chat-v3.1:free","max_tokens":100}'
```

**3) Генерация в стриминге (SSE)**

```bash
curl -N -sS -X POST "http://127.0.0.1:8000/generate?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Дай пошаговое решение 1+1, стримом","model":"moonshotai/kimi-k2:free","max_tokens":50}'
```

> Флаг `-N` (no-buffer) позволяет видеть приходящие SSE-ивенты сразу.

**4) Benchmark (multipart/form-data)**

```bash
curl -sS -X POST "http://127.0.0.1:8000/benchmark" \
  -F "prompt_file=@prompts.txt;type=text/plain" \
  -F "model=deepseek/deepseek-chat-v3.1:free" \
  -F "runs=3"
```

**5) Benchmark с визуализацией (HTML-таблица)**

```bash
curl -sS -X POST "http://127.0.0.1:8000/benchmark?visualize=true" \
  -F "prompt_file=@prompts.txt;type=text/plain" \
  -F "model=moonshotai/kimi-k2:free" \
  -F "runs=5"
```

**6) Измерение полной latency (в секундах) для `/generate`**

```bash
curl -o /dev/null -sS -w "time_total=%{time_total}\n" \
  -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Скажи привет","model":"deepseek/deepseek-chat-v3.1:free"}'
```

**7) Измерение полной latency для `/benchmark` (только время запроса)**

```bash
curl -o /dev/null -sS -w "time_total=%{time_total}\n" \
  -X POST "http://127.0.0.1:8000/benchmark" \
  -F "prompt_file=@prompts.txt;type=text/plain" \
  -F "model=deepseek/deepseek-chat-v3.1:free" \
  -F "runs=3"
```
