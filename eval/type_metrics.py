"""python -m eval.type_metrics labels.csv labels_types.csv cache_dir [cache_boxes_dir]
Метрики по ТИПАМ нарушений, а не «поймал ли что-нибудь».

Положительные для правила — только видео этого типа (по data/labels_types.csv);
видео другого типа и чистая норма считаются отрицательными: срабатывание там —
ложная тревога, даже если нарушение в ролике есть.
Строки с type=unclear из метрик исключаются и печатаются отдельно.
"""
import csv
import sys
from pathlib import Path

from app.rules import ENGINE_RULES, RULES
from eval.evaluate import events_by_rule, gold_times


def load_types(path):
    return {r["video"]: [t.strip() for t in (r.get("type") or "").split(";") if t.strip()]
            for r in csv.DictReader(open(path, encoding="utf-8"))}


def main(labels, types_csv, cache_dir="cache", boxes_dir="cache_boxes"):
    types = load_types(types_csv)
    rows = []
    for row in csv.DictReader(open(labels, encoding="utf-8")):
        name = row["video"]
        if not ((Path(cache_dir) / f"{name}.json").exists() or (Path(boxes_dir) / f"{name}.json").exists()):
            continue
        rows.append((row, types.get(name, []), events_by_rule(name, cache_dir, boxes_dir)))

    known = [(r, t, e) for r, t, e in rows if t and "unclear" not in t]
    unclear = [r["video"] for r, t, e in rows if "unclear" in t]
    print(f"видео: {len(rows)}, с известным типом: {len(known)}, unclear: {len(unclear)} {unclear}")
    print(f"чистая норма: {sum(1 for r, t, e in rows if r['label'] == 'no_violation')}\n")

    print(f"{'правило':38} {'своих':>6} {'TP':>3} {'FN':>3} {'FP(чужой тип)':>13} {'FP(норма)':>9}")
    for rid, rule in ENGINE_RULES.items():
        own = [(r, e) for r, t, e in known if rid in t]
        tp = sum(1 for _, e in own if e.get(rid))
        fn = len(own) - tp
        fp_other = sum(1 for r, t, e in known if rid not in t and e.get(rid))
        fp_clean = sum(1 for r, t, e in rows if r["label"] == "no_violation" and e.get(rid))
        print(f"{rid:38} {len(own):6} {tp:3} {fn:3} {fp_other:13} {fp_clean:9}")

    print("\nошибки таймкода по своим видео:")
    for rid in ENGINE_RULES:
        errs = []
        for r, t, e in known:
            if rid in t and e.get(rid):
                for g in gold_times(r):
                    errs.append(min(abs(x["t"] - g) for x in e[rid]))
        if errs:
            print(f"  {rid}: среднее {sum(errs)/len(errs):.1f} с, максимум {max(errs):.1f} с")

    covered = set()
    for r, t, e in known:
        for rid in t:
            if e.get(rid):
                covered.add(r["video"])
    total = {r["video"] for r, t, e in known}
    print(f"\nпокрытие таксономии хоть каким-то правилом: {len(covered)} из {len(total)}")
    print("не покрыты:", sorted(total - covered, key=lambda s: int(s.split('-')[1])))
    absent = sorted({rid for r, t, e in known for rid in t} - set(ENGINE_RULES))
    if absent:
        print("типы без правила в rules.json:", absent)
    print("подписи правил:", {rid: RULES[rid]["type"][:40] for rid in ENGINE_RULES})


if __name__ == "__main__":
    main(*sys.argv[1:])
