"""Признак «кассир на рабочем месте» по боксам людей, без VLM.

Почему так: место кассира неподвижно, а сам он сидит на нём почти весь клип.
Опорный прямоугольник берётся из первых секунд (кассир за столом) или, если там
пусто, из самого частого положения человека. Кадр считается «кассир на месте»,
если центр бокса человека попадает в этот прямоугольник, расширенный на dx/dy.

В отличие от предиката VLM (AUC 0.61 на 42 видео) признак детерминирован,
считается за секунды и не требует GPU-VLM.

Модуль чистый: на вход — времена и боксы из детектора, без torch.
"""
from typing import Dict, List, Optional, Sequence

Box = Sequence[float]  # x1, y1, x2, y2


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def center(b: Box) -> tuple:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _median_box(boxes: List[Box]) -> List[float]:
    return [sorted(b[i] for b in boxes)[len(boxes) // 2] for i in range(4)]


def densest_cluster(detections: List[List[Box]], cell: float = 0.12) -> List[Box]:
    """Боксы самого частого положения человека (сетка по центру бокса).

    Нужен, когда в начале клипа кассира в кадре нет (например X-31): опорную
    точку берём там, где человек проводит больше всего времени.
    """
    cells: Dict[tuple, List[Box]] = {}
    for boxes in detections:
        for b in boxes:
            cx, cy = center(b)
            key = (int(cx // (cell * 1000)), int(cy // (cell * 1000)))
            cells.setdefault(key, []).append(tuple(b[:4]))
    if not cells:
        return []
    return [list(b) for b in max(cells.values(), key=len)]


def seat_box(times: List[float], detections: List[List[Box]],
             warmup: float = 2.0) -> Optional[List[float]]:
    early = [tuple(b[:4]) for t, boxes in zip(times, detections)
             if t <= warmup for b in boxes]
    pool = early if len(early) >= 2 else densest_cluster(detections)
    return _median_box(pool) if pool else None


def zone_of(seat: Box, dx: float = 0.8, dy: float = 0.4) -> List[float]:
    """Рабочее место: прямоугольник кассира, расширенный на доли его ширины/высоты.

    Небольшое расширение (0.2) ловит «встал и отошёл на шаг», большое (0.8)
    пропускает подъём к сейфу и ловит только реальный уход из зоны.
    """
    w, h = seat[2] - seat[0], seat[3] - seat[1]
    return [seat[0] - dx * w, seat[1] - dy * h, seat[2] + dx * w, seat[3] + dy * h]


def present(box: Box, zone: Box) -> bool:
    cx, cy = center(box)
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def timeline(times: List[float], detections: List[List[Box]], dx: float = 0.8,
             dy: float = 0.4, warmup: float = 2.0) -> List[dict]:
    """[{"t": сек, "p": {"cashier_present": 1.0|0.0}}] на той же сетке времён."""
    seat = seat_box(times, detections, warmup)
    out = []
    for t, boxes in zip(times, detections):
        ok = seat is not None and any(present(b, zone_of(seat, dx, dy)) for b in boxes)
        out.append({"t": t, "p": {"cashier_present": 1.0 if ok else 0.0}})
    return out
