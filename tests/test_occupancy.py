"""Тесты признака «кассир за столом». Без GPU и torch: только чистые функции."""
import json
from pathlib import Path

from app.engine import find_events
from app.occupancy import densest_cluster, iou, present, seat_box, timeline, zone_of

SEAT = [100, 200, 200, 400]


def test_iou_identical_and_disjoint():
    assert iou(SEAT, SEAT) == 1.0
    assert iou(SEAT, [500, 500, 600, 600]) == 0.0
    assert 0 < iou(SEAT, [150, 300, 250, 500]) < 1


def test_seat_box_uses_first_seconds():
    times = [0.0, 0.5, 1.0, 1.5, 5.0]
    dets = [[[100, 200, 200, 400]], [[102, 202, 202, 400]], [[104, 198, 198, 400]],
            [[98, 200, 200, 402]], [[0, 0, 50, 90]]]
    assert seat_box(times, dets) == [102, 200, 200, 400]


def test_seat_box_falls_back_to_densest_cluster():
    # в начале клипа никого нет — опора берётся там, где человек проводит время
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    dets = [[], [], [[100, 200, 200, 400]], [[101, 201, 201, 401]],
            [[103, 203, 203, 403]], [[104, 204, 204, 404]]]
    seat = seat_box(times, dets)
    assert seat is not None and abs(seat[0] - 101) < 4
    cluster = densest_cluster(dets)
    assert cluster and abs(sum(b[0] for b in cluster) / len(cluster) - 101) < 5


def test_zone_expands_seat():
    z = zone_of(SEAT, 0.5, 0.25)
    assert z == [50, 150, 250, 450]


def test_present_uses_box_center():
    z = zone_of(SEAT, 0.2, 0.1)
    assert present([90, 210, 110, 230], z)          # центр внутри
    assert not present([400, 210, 420, 230], z)     # центр снаружи


def test_timeline_marks_absence():
    times = [0.0, 1.0, 2.0, 3.0]
    dets = [[[100, 200, 200, 400]], [[100, 200, 200, 400]], [], []]
    tl = timeline(times, dets)
    assert [r["p"]["cashier_present"] for r in tl] == [1.0, 1.0, 0.0, 0.0]


def test_long_absence_breaks_rule_engine():
    cfg = {r["id"]: r for r in json.loads(Path("rules.json").read_text(encoding="utf-8"))["violation_types"]}
    leave = cfg["cashier_left_workplace"]["engine"]
    assert leave["when"]["pred"] == "cashier_present"
    times = [i * 0.5 for i in range(20)]
    dets = [[[100, 200, 200, 400]]] * 8 + [[]] * 6 + [[[100, 200, 200, 400]]] * 6
    ev = find_events(timeline(times, dets), leave)
    assert len(ev) == 1 and abs(ev[0]["t"] - 4.0) < 1e-6, ev


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
