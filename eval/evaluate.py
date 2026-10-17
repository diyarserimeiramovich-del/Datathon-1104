"""python -m eval.evaluate labels.csv cache_dir [cache_boxes_dir]
Метрики по закэшированным сигналам (движок правил из rules.json, только enabled=true).

Источник сигнала на правило:
- detector="seat_occupancy" -> боксы людей из cache_boxes (см. eval/detect_boxes.py);
- иначе                     -> предикаты VLM из cache.
Формат меток: video,label,timecode ИЛИ video,...,label,timecodes_sec (через ';').
Для видео с несколькими нарушениями (X-4 / X-47) ошибка таймкода считается для каждого эталонного таймкода.
"""
import csv
import json
import sys
from pathlib import Path

from app.engine import find_events
from app.occupancy import timeline as occ_timeline
from app.rules import ENGINE_RULES


def gold_times(row):
    raw = row.get("timecodes_sec") or row.get("timecode") or ""
    return [float(x) for x in str(raw).split(";") if x.strip()]


def load_boxes(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["frames"]
    return [r["t"] for r in rows], [r["boxes"] for r in rows]


def events_by_rule(name, cache_dir, boxes_dir):
    """События по каждому правилу для одного видео."""
    vlm_path = Path(cache_dir) / f"{name}.json"
    vlm = json.loads(vlm_path.read_text(encoding="utf-8"))["timeline"] if vlm_path.exists() else None
    box_path = Path(boxes_dir) / f"{name}.json"
    boxes = load_boxes(box_path) if box_path.exists() else None
    out = {}
    for rid, rule in ENGINE_RULES.items():
        if rule.get("detector") == "seat_occupancy":
            if boxes is None:
                continue
            z = rule.get("zone") or {}
            tl = occ_timeline(boxes[0], boxes[1], z.get("dx", 0.8), z.get("dy", 0.4))
        else:
            if vlm is None:
                continue
            tl = vlm
        out[rid] = find_events(tl, rule["engine"])
    return out


def main(labels, cache_dir, boxes_dir="cache_boxes"):
    tp = fp = tn = fn = 0
    errs = []
    per_rule = {rid: [0, 0] for rid in ENGINE_RULES}   # [TP, FP]
    for row in csv.DictReader(open(labels, encoding="utf-8")):
        name = row["video"]
        if not (Path(cache_dir) / f"{name}.json").exists() and \
                not (Path(boxes_dir) / f"{name}.json").exists():
            print("нет кэша:", name)
            continue
        evs = [(rid, e) for rid, es in events_by_rule(name, cache_dir, boxes_dir).items() for e in es]
        pred, gold = ("violation" if evs else "no_violation"), row["label"]
        for rid, _ in evs:
            per_rule[rid][0 if gold == "violation" else 1] += 1
        if gold == "violation" and pred == "violation":
            tp += 1
            for g in gold_times(row):
                errs.append(min(abs(e["t"] - g) for _, e in evs))
        elif gold == "violation":
            fn += 1
            print("ПРОПУСК:", name, "метка", row.get("timecodes_sec") or row.get("timecode"))
        elif pred == "violation":
            fp += 1
            print("ЛОЖНОЕ СРАБАТЫВАНИЕ:", name, [(r, round(e["t"], 1)) for r, e in evs])
        else:
            tn += 1
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    for rid, (t, f) in per_rule.items():
        if t or f:
            print(f"  {rid}: TP={t} FP={f}")
    if errs:
        print(f"Ошибка таймкода: среднее {sum(errs)/len(errs):.1f} с, максимум {max(errs):.1f} с")


if __name__ == "__main__":
    main(*sys.argv[1:])
