"""python -m eval.sweep labels.csv cache_dir [rule_id] [lo] [hi] [step] [field] [cache_boxes]
Подбор одного числового параметра правила по всему кэшу. Остальные включённые правила
учитываются как есть, поэтому TP/FP/TN/FN совпадают с eval/evaluate.py на выбранном значении.

field: gte / lt (порог предиката VLM) либо min_duration_sec (длительность для сенсора).
По умолчанию: gte для правил на предикатах VLM, min_duration_sec для detector=seat_occupancy.
В конце печатается точка «0 ложных, затем максимум TP» и подробности по ней.
"""
import copy
import csv
import json
import sys
from pathlib import Path

from app.engine import find_events
from app.occupancy import timeline as occ_timeline
from app.rules import ENGINE_RULES
from eval.evaluate import gold_times, load_boxes


def load(labels, cache_dir, boxes_dir):
    rows, missing = [], []
    for row in csv.DictReader(open(labels, encoding="utf-8")):
        name = row["video"]
        vlm_p = Path(cache_dir) / f"{name}.json"
        box_p = Path(boxes_dir) / f"{name}.json"
        if not vlm_p.exists() and not box_p.exists():
            missing.append(name)
            continue
        rows.append((row,
                     json.loads(vlm_p.read_text(encoding="utf-8"))["timeline"] if vlm_p.exists() else None,
                     load_boxes(box_p) if box_p.exists() else None))
    return rows, missing


def _source(rule, vlm_tl, boxes):
    if rule.get("detector") == "seat_occupancy":
        if boxes is None:
            return None
        z = rule.get("zone") or {}
        return occ_timeline(boxes[0], boxes[1], z.get("dx", 0.8), z.get("dy", 0.4))
    return vlm_tl


def events(rows, rule_id, field, value):
    target = copy.deepcopy(ENGINE_RULES[rule_id])
    if field in ("gte", "lt"):
        target["engine"]["when"][field] = value
    else:
        target["engine"][field] = value
    out = []
    for row, vlm_tl, boxes in rows:
        tl = _source(target, vlm_tl, boxes)
        evs = [e for rid, r in ENGINE_RULES.items() if rid != rule_id
               for e in find_events(_source(r, vlm_tl, boxes) or [], r["engine"])]
        if tl is not None:
            evs += find_events(tl, target["engine"])
        out.append((row, sorted(evs, key=lambda e: e["t"])))
    return out


def score(rows, rule_id, field, value):
    tp = fp = tn = fn = 0
    errs, fps, misses = [], [], []
    for row, evs in events(rows, rule_id, field, value):
        gold = row["label"] == "violation"
        if gold and evs:
            tp += 1
            for g in gold_times(row):
                errs.append(min(abs(e["t"] - g) for e in evs))
        elif gold:
            fn += 1
            misses.append(row["video"])
        elif evs:
            fp += 1
            fps.append((row["video"], [round(e["t"], 1) for e in evs]))
        else:
            tn += 1
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, errs=errs, fps=fps, misses=misses)


def main(labels, cache_dir, rule_id="cashier_left_workplace", lo=0.30, hi=0.80, step=0.01,
         field=None, boxes_dir="cache_boxes"):
    rows, missing = load(labels, cache_dir, boxes_dir)
    if field is None:
        field = "min_duration_sec" if ENGINE_RULES[rule_id].get("detector") == "seat_occupancy" else "gte"
    print(f"видео с кэшем: {len(rows)}, без кэша: {len(missing)}; правило {rule_id}, параметр {field}")
    grid, g = [], float(lo)
    while g <= float(hi) + 1e-9:
        v = round(g, 3)
        grid.append((v, score(rows, rule_id, field, v)))
        g = round(g + float(step), 3)
    print("\nзначение  TP  FP  TN  FN точн полн  ср.с макс.с")
    for v, s in grid:
        prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
        mean = f"{sum(s['errs'])/len(s['errs']):.1f}" if s["errs"] else "-"
        mx = f"{max(s['errs']):.1f}" if s["errs"] else "-"
        print(f"{v:8.2f} {s['tp']:3d} {s['fp']:3d} {s['tn']:3d} {s['fn']:3d} {prec:5.2f} {rec:5.2f} {mean:>5} {mx:>6}")
    clean = [(v, s) for v, s in grid if s["fp"] == 0]
    if clean:
        best_tp = max(s["tp"] for _, s in clean)
        cand = [(v, s) for v, s in clean if s["tp"] == best_tp]
        print(f"\nВыбор: {field}={min(cand, key=lambda x: x[0])[0]:.2f} -> "
              f"TP={best_tp} FP=0 (из {len(cand)} равных точек)")
        print("пропуски:", cand[0][1]["misses"])
    print("нет кэша:", missing)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *a[2:3], *([float(x) for x in a[3:6]] or []),
         *([a[6]] if len(a) > 6 else []))
