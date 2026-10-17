"""Темпоральный движок правил.

Вход: timeline = [{"t": сек, "p": {предикат: вероятность}}, ...]
Выход: события по конфигу правила из rules.json (поле "engine").
Модуль не импортирует тяжёлых библиотек, поэтому тестируется без GPU.
"""
from typing import Dict, List, Set


def holds(cond: dict, p: Dict[str, float]) -> bool:
    if "all" in cond:
        return all(holds(c, p) for c in cond["all"])
    if "any" in cond:
        return any(holds(c, p) for c in cond["any"])
    if "not" in cond:
        return not holds(cond["not"], p)
    v = p.get(cond["pred"])
    if v is None:          # нет данных: консервативно считаем условие невыполненным
        return False
    if "gte" in cond and not v >= cond["gte"]:
        return False
    if "lt" in cond and not v < cond["lt"]:
        return False
    return True


def preds_in(cond: dict) -> Set[str]:
    if "all" in cond or "any" in cond:
        return set().union(*(preds_in(c) for c in cond.get("all", cond.get("any"))))
    if "not" in cond:
        return preds_in(cond["not"])
    return {cond["pred"]}


def _score(cond: dict, p: Dict[str, float]) -> float:
    vals: List[float] = []

    def walk(c):
        if "all" in c or "any" in c:
            for x in c.get("all", c.get("any")):
                walk(x)
        elif "not" in c:
            return
        else:
            v = p.get(c["pred"], 0.0)
            vals.append(v if "gte" in c else 1.0 - v)

    walk(cond)
    return sum(vals) / len(vals) if vals else 0.5


def _runs(flags: List[bool], max_gap: int):
    """Непрерывные отрезки True; провалы не длиннее max_gap склеиваются."""
    runs, start, last, gap = [], None, None, 0
    for i, f in enumerate(flags):
        if f:
            if start is None:
                start = i
            last, gap = i, 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, last))
                start, gap = None, 0
    if start is not None:
        runs.append((start, last))
    return runs


def find_events(timeline: List[dict], cfg: dict) -> List[dict]:
    tl = sorted(timeline, key=lambda r: r["t"])
    gap = cfg.get("max_gap_samples", 1)
    events: List[dict] = []

    if cfg.get("type", "sustained") == "sustained":
        flags = [holds(cfg["when"], r["p"]) for r in tl]
        for a, b in _runs(flags, gap):
            dur = tl[b]["t"] - tl[a]["t"]
            if (b - a + 1) < cfg.get("min_samples", 2) or dur < cfg.get("min_duration_sec", 0.0):
                continue
            peak = max(range(a, b + 1), key=lambda i: _score(cfg["when"], tl[i]["p"]))
            t = tl[a]["t"] if cfg.get("anchor", "start") == "start" else tl[peak]["t"]
            events.append({"t": t, "start": tl[a]["t"], "end": tl[b]["t"],
                           "peak": tl[peak]["t"], "score": _score(cfg["when"], tl[peak]["p"])})

    elif cfg["type"] == "sequence":
        base = [holds(cfg["base"], r["p"]) for r in tl]
        trig = [holds(cfg["trigger"], r["p"]) for r in tl]
        need = cfg.get("base_min_duration_sec", 0.0)
        k = cfg.get("trigger_min_samples", 2)
        for a, b in _runs(base, gap):
            t0 = tl[a]["t"]
            idx = [i for i in range(a, b + 1) if tl[i]["t"] >= t0 + need]
            for j in range(len(idx) - k + 1):
                if all(trig[idx[j + m]] for m in range(k)):
                    ti = idx[j]
                    t = tl[ti]["t"] if cfg.get("anchor", "trigger") == "trigger" else t0
                    events.append({"t": t, "start": t0, "end": tl[b]["t"], "peak": tl[ti]["t"],
                                   "score": _score(cfg["base"], tl[ti]["p"])})
                    break
    else:
        raise ValueError(f"unknown engine type: {cfg['type']}")

    events.sort(key=lambda e: e["t"])
    out: List[dict] = []
    for e in events:
        if out and e["t"] - out[-1]["t"] < cfg.get("cooldown_sec", 10.0):
            continue
        out.append(e)
    return out


def rule_preds(cfg: dict) -> Set[str]:
    conds = [cfg[k] for k in ("when", "base", "trigger") if k in cfg]
    return set().union(*(preds_in(c) for c in conds)) if conds else set()
