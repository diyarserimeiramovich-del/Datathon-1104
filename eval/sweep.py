"""python -m eval.sweep labels.csv cache_dir [rule_id] [lo] [hi] [step]
Подбор порога gte у одного правила по всему кэшу. Остальные включённые правила
учитываются как есть, поэтому TP/FP/TN/FN совпадают с eval/evaluate.py на выбранном пороге.
Печатает сетку и подробности по видео для порога с 0 FP и максимальным TP."""
import copy
import csv
import json
import sys
from pathlib import Path

from app.engine import find_events
from app.rules import ENGINE_RULES
from eval.evaluate import gold_times


def load(labels, cache_dir):
    rows, missing = [], []
    for row in csv.DictReader(open(labels, encoding="utf-8")):
        p = Path(cache_dir) / f"{row['video']}.json"
        if not p.exists():
            missing.append(row["video"])
            continue
        rows.append((row, json.loads(p.read_text(encoding="utf-8"))["timeline"]))
    return rows, missing


def events(rows, rule_id, gte):
    cfg = copy.deepcopy(ENGINE_RULES[rule_id]["engine"])
    cfg["when"]["gte"] = gte
    others = {k: r["engine"] for k, r in ENGINE_RULES.items() if k != rule_id}
    out = []
    for row, tl in rows:
        evs = [e for cfg_o in others.values() for e in find_events(tl, cfg_o)]
        evs += find_events(tl, cfg)
        out.append((row, sorted(evs, key=lambda e: e["t"])))
    return out


def score(rows, rule_id, gte):
    tp = fp = tn = fn = 0
    errs, fps, misses = [], [], []
    for row, evs in events(rows, rule_id, gte):
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


def main(labels, cache_dir, rule_id="cashier_left_workplace", lo=0.30, hi=0.80, step=0.01):
    rows, missing = load(labels, cache_dir)
    print(f"видео с кэшем: {len(rows)}, без кэша: {len(missing)}")
    grid = []
    g = round(lo, 3)
    while g <= hi + 1e-9:
        s = score(rows, rule_id, round(g, 3))
        grid.append((round(g, 3), s))
        g = round(g + step, 3)
    print("\n порог  TP  FP  TN  FN точн полн  ср.с макс.с")
    for gte, s in grid:
        prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
        mean = f"{sum(s['errs'])/len(s['errs']):.1f}" if s["errs"] else "-"
        mx = f"{max(s['errs']):.1f}" if s["errs"] else "-"
        print(f"{gte:6.2f} {s['tp']:3d} {s['fp']:3d} {s['tn']:3d} {s['fn']:3d} {prec:6.2f} {rec:6.2f} {mean:>5} {mx:>6}")
    clean = [(g, s) for g, s in grid if s["fp"] == 0]
    if clean:
        best_tp = max(s["tp"] for _, s in clean)
        cand = [(g, s) for g, s in clean if s["tp"] == best_tp]
        mean_err = lambda s: sum(s["errs"]) / len(s["errs"]) if s["errs"] else 9e9
        gte, s = min(cand, key=lambda x: mean_err(x[1]))
        print(f"\nВыбор: gte={gte:.2f} -> TP={s['tp']} FP={s['fp']} TN={s['tn']} FN={s['fn']}")
        print("пропуски:", s["misses"])
        print("ошибки таймкода:", [round(e, 1) for e in s["errs"]])
    print("нет кэша:", missing)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2] if len(a) > 2 else "cashier_left_workplace",
         *([float(x) for x in a[3:6]] or []))
