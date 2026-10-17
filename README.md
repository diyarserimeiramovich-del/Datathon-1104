# Cashier discipline AI

## Запуск без моделей (проверка API и интерфейса)
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    API_KEY=ключ uvicorn app.main:app --host 0.0.0.0 --port 8000
UI: http://localhost:8000 · API: `POST /api/v1/analyze`, `GET /api/v1/jobs/{id}` (заголовок `X-API-Key`).
Без `USE_VLM=1` сервис отвечает «нарушение не обнаружено» на всё.

## Запуск с VLM (нужен GPU)
    pip install -r requirements-vlm.txt
    USE_VLM=1 MODEL_ID=Qwen/Qwen2-VL-2B-Instruct API_KEY=ключ uvicorn app.main:app --port 8000
Для работы без интернета скачайте веса заранее и укажите локальный путь в `MODEL_ID` и `HF_HUB_OFFLINE=1`.

## Тесты и оценка
    python -m tests.test_engine                                   # логика правил, без GPU
    python -m eval.labels open_labels.xlsx > data/labels.csv      # метки (уже лежит в data/)
    python -m eval.cache_predicates videos/ cache/                # тяжёлый проход один раз
    python -m eval.evaluate data/labels.csv cache/                # пороги в rules.json меняйте и пересчитывайте

## Где что
- `rules.json` типы нарушений, пороги, проверенные ссылки на нормы
- `predicates.json` вопросы к VLM
- `app/engine.py` правила во времени · `app/predicates.py` VLM · `app/detectors.py` связка
- `docs/` разбор данных и нормативка

## Docker
- `Dockerfile` (CPU, без модели) отвечает «нарушение не обнаружено» на всё: годится только проверить API.
- `Dockerfile.gpu` ставит VLM, включает `USE_VLM=1` и офлайн-режим; веса монтируются с диска (инструкция в конце файла).
- Устройство выбирается автоматически (`DEVICE=cuda|cpu`); на CPU 2B-модель слишком медленна для лимита 5 минут.
