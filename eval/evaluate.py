"""python -m eval.evaluate labels.csv cache_dir
Метрики по закэшированным предикатам (движок правил из rules.json, только enabled=true).
Формат меток: video,label,timecode  ИЛИ  video,...,label,timecodes_sec (несколько через ';', см. data/labels_grouped.csv).
Для видео с несколькими нарушениями (X-4 / X-47) ошибка таймкода считается для каждого эталонного таймкода."""
import csv
import json
import sys
from pathlib import Path

from app.engine import find_events
from app.rules import ENGINE_RULES


def gold_times(row):
    raw = row.get("timecodes_sec") or row.get("timecode") or ""
    return [float(x) for x in str(raw).split(";") if x.strip()]


def main(labels, cache_dir):
    tp = fp = tn = fn = 0
    errs = []
    for row in csv.DictReader(open(labels, encoding="utf-8")):
        p = Path(cache_dir) / f"{row['video']}.json"
        if not p.exists():
            print("нет кэша:", row["video"])
            continue
        tl = json.loads(p.read_text(encoding="utf-8"))["timeline"]
        evs = [(rid, e) for rid, r in ENGINE_RULES.items() for e in find_events(tl, r["engine"])]
        pred, gold = ("violation" if evs else "no_violation"), row["label"]
        if gold == "violation" and pred == "violation":
            tp += 1
            for g in gold_times(row):
                errs.append(min(abs(e["t"] - g) for _, e in evs))
        elif gold == "violation":
            fn += 1
            print("ПРОПУСК:", row["video"], "метка", row.get("timecodes_sec") or row.get("timecode"))
        elif pred == "violation":
            fp += 1
            print("ЛОЖНОЕ СРАБАТЫВАНИЕ:", row["video"], [(r, round(e["t"], 1)) for r, e in evs])
        else:
            tn += 1
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    if errs:
        print(f"Ошибка таймкода: среднее {sum(errs)/len(errs):.1f} с, максимум {max(errs):.1f} с")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
