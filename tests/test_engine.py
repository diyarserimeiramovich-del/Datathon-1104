"""Запуск: python -m tests.test_engine   (без GPU и моделей)"""
import json
from pathlib import Path

from app.engine import find_events, holds

CFG = {r["id"]: r["engine"] for r in json.loads(Path("rules.json").read_text(encoding="utf-8"))["violation_types"]
       if r.get("engine")}["cash_left_unsecured"]


LEAVE = {r["id"]: r["engine"] for r in json.loads(Path("rules.json").read_text(encoding="utf-8"))["violation_types"]
         if r.get("engine")}["cashier_left_workplace"]


def make(fn, t_end=72, step=1.0):
    return [{"t": t * step, "p": fn(t * step)} for t in range(int(t_end / step) + 1)]


def x4(t):  # как в Х-4: деньги остаются на столе после ухода клиента, входит следующий
    return {"client_seated": 0.9 if 12 <= t <= 49 else 0.1,
            "cash_on_desk": 0.9 if t >= 14 else 0.1,
            "visitor_present": 0.9 if (8 <= t <= 11 or 50 <= t <= 53 or 56 <= t <= 62) else 0.1}


def x5(t):  # как в Х-5: деньги убраны в сейф до ухода клиента
    return {"client_seated": 0.9 if 12 <= t <= 55 else 0.1,
            "cash_on_desk": 0.9 if 14 <= t <= 48 else 0.1,
            "visitor_present": 0.9 if (4 <= t <= 8 or 56 <= t <= 70) else 0.1}


def test_x4_detected_near_label():
    ev = find_events(make(x4), CFG)
    assert len(ev) == 1, ev
    assert abs(ev[0]["t"] - 57) <= 2, ev   # метка в xlsx: 57 с


def test_x5_no_violation():
    assert find_events(make(x5), CFG) == []


def test_single_frame_spike_ignored():
    tl = make(lambda t: {"client_seated": 0.1, "cash_on_desk": 0.9 if t == 20 else 0.1, "visitor_present": 0.9})
    assert find_events(tl, CFG) == []


def test_departing_client_alone_is_not_trigger():
    # клиент уходит (виден как «стоит/идёт»), деньги ещё на столе, но нового человека нет
    def f(t):
        return {"client_seated": 0.9 if t <= 49 else 0.1, "cash_on_desk": 0.9,
                "visitor_present": 0.9 if 50 <= t <= 53 else 0.1}
    assert find_events(make(f), CFG) == []


def test_missing_predicate_is_false():
    assert holds({"pred": "x", "gte": 0.5}, {}) is False


def test_cashier_leaves_room_detected():
    # как в X-26: кассир за столом до 9 с, потом уходит; клиента нет
    tl = make(lambda t: {"cashier_present": 0.05 if t >= 10 else 0.95}, t_end=19)
    ev = find_events(tl, LEAVE)
    assert len(ev) == 1 and abs(ev[0]["t"] - 10) <= 1, ev


def test_short_absence_ignored():
    tl = make(lambda t: {"cashier_present": 0.05 if t == 5 else 0.95}, t_end=20)
    assert find_events(tl, LEAVE) == []


def test_cashier_always_present_no_event():
    tl = make(lambda t: {"cashier_present": 0.95}, t_end=20)
    assert find_events(tl, LEAVE) == []


def test_absence_shorter_than_min_duration_ignored():
    # провал 2 с короче min_duration_sec=2.5 из rules.json
    tl = make(lambda t: {"cashier_present": 0.05 if 5 <= t <= 7 else 0.95}, t_end=20)
    assert find_events(tl, LEAVE) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
