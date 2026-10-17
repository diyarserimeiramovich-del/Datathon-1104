# Cashier discipline AI

## Запуск без моделей (проверка API и интерфейса)
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    API_KEY=ключ uvicorn app.main:app --host 0.0.0.0 --port 8000
UI: http://localhost:8000 · API: `POST /api/v1/analyze`, `GET /api/v1/jobs/{id}` (заголовок `X-API-Key`).
Без torch/torchvision и без `USE_VLM=1` сервис отвечает «нарушение не обнаружено» на всё.

## Признак «кассир за столом» (по умолчанию, без VLM)
Правило «покинул рабочее место» считается по детектору людей: torchvision FasterRCNN-MobileNet,
зона рабочего места из первых секунд клипа. Нужен только torch+torchvision (`requirements-vlm.txt`);
если их нет, правило отключается само, остальное работает. Выключить явно: `USE_OCCUPANCY=0`.
По размеченным типам ловит 3 из 6 роликов «ушёл с рабочего места» без ложных на норме;
выигрыш над VLM — одно видео, но в 60 раз быстрее. Подробности и оговорки — `docs/06_OCCUPANCY_SENSOR.md`.

## Запуск с VLM (нужен GPU)
    pip install -r requirements-vlm.txt
    USE_VLM=1 MODEL_ID=Qwen/Qwen2-VL-2B-Instruct API_KEY=ключ uvicorn app.main:app --port 8000
Для работы без интернета скачайте веса заранее и укажите локальный путь в `MODEL_ID` и `HF_HUB_OFFLINE=1`.

## Тесты и оценка
    python -m tests.test_engine                                   # логика правил, без GPU
    python -m eval.labels open_labels.xlsx > data/labels.csv      # метки (уже лежит в data/)
    python -m eval.detect_boxes videos/ cache_boxes/              # боксы людей для сенсора, ~2 мин на 42 видео
    python -m eval.cache_predicates videos/ cache/                # тяжёлый проход VLM, ~2 ч на 42 видео
    python -m eval.evaluate data/labels_grouped.csv cache/ cache_boxes/   # метрики по обоим сигналам
    python -m eval.sweep data/labels_grouped.csv cache/           # сетка параметров по всему кэшу
    python -m eval.type_metrics data/labels_grouped.csv data/labels_types.csv cache/ cache_boxes/  # метрики по типам
    python -m tests.test_occupancy                                # признак занятости, без GPU
    python -m eval.contact_sheets data/labels_grouped.csv videos/  # кадры вокруг меток для ручной разметки типов

## Где что
- `rules.json` типы нарушений, пороги, проверенные ссылки на нормы
- `predicates.json` вопросы к VLM
- `app/engine.py` правила во времени · `app/predicates.py` VLM · `app/occupancy.py` признак «кассир за столом» · `app/detectors.py` связка
- `docs/` разбор данных и нормативка

## Docker
- `Dockerfile` (CPU, без модели) отвечает «нарушение не обнаружено» на всё: годится только проверить API.
- `Dockerfile.gpu` ставит VLM, включает `USE_VLM=1` и офлайн-режим; веса монтируются с диска (инструкция в конце файла).
- Устройство выбирается автоматически (`DEVICE=cuda|cpu`); на CPU 2B-модель слишком медленна для лимита 5 минут.
