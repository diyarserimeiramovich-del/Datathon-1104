"""Детекторы. Контракт: detector.run(frames) -> list[Event]."""
import math
import os
import time
from dataclasses import dataclass, field
from typing import List

from .engine import find_events, rule_preds
from .video import Frame


@dataclass
class Event:
    rule_id: str
    t: float                      # таймкод события, сек
    description: str              # что видно и почему это может быть нарушением
    frames: List[Frame] = field(default_factory=list)  # 1–3 кадра-доказательства
    score: float = 0.0


class Detector:
    name = "base"

    def run(self, frames: List[Frame]) -> List[Event]:
        raise NotImplementedError


class NullDetector(Detector):
    """Заглушка: ничего не находит."""
    name = "null"

    def run(self, frames):
        return []


def _nearest(frames: List[Frame], t: float) -> Frame:
    return min(frames, key=lambda f: abs(f.t - t))


def pick_evidence(frames: List[Frame], start: float, peak: float) -> List[Frame]:
    chosen, seen = [], set()
    for t in (start, peak, start + 1.0):
        f = _nearest(frames, t)
        if id(f) not in seen:
            seen.add(id(f))
            chosen.append(f)
    return sorted(chosen, key=lambda f: f.t)[:3]


class RuleEngineDetector(Detector):
    """Предикаты VLM по кадрам, затем правила из rules.json.

    1) грубый проход по сетке кадров (шаг подбирается под бюджет проходов модели),
    2) для найденных событий точный проход вокруг момента (шаг 0.5 с), чтобы уточнить таймкод.
    """
    name = "rule_engine"

    def __init__(self, extractor, rules: dict, spec_budget_sec: float = 200.0,
                 max_passes: int = int(os.environ.get("MAX_PASSES", "240"))):
        self.extractor, self.rules = extractor, rules
        self.budget, self.max_passes = spec_budget_sec, max_passes

    def run(self, frames: List[Frame]) -> List[Event]:
        if not frames or not self.rules:
            return []
        deadline = time.time() + self.budget
        self.extractor.reset()
        all_preds = sorted(set().union(*(rule_preds(r["engine"]) for r in self.rules.values())))
        stride = max(1, math.ceil(len(frames) * len(all_preds) / self.max_passes))
        coarse = frames[::stride]
        step = (coarse[1].t - coarse[0].t) if len(coarse) > 1 else 0.5
        coarse_tl = self.extractor.timeline(coarse, all_preds, deadline)

        events: List[Event] = []
        for rid, rule in self.rules.items():
            cfg, preds = rule["engine"], sorted(rule_preds(rule["engine"]))
            for ev in find_events(coarse_tl, cfg):
                lo, hi = ev["start"] - step, ev["end"] + step
                fine = [f for f in frames if lo - 1e-6 <= f.t <= hi + 1e-6]
                fine_tl = self.extractor.timeline(fine, preds, deadline)
                merged = {round(r["t"], 2): r for r in coarse_tl}
                merged.update({round(r["t"], 2): r for r in fine_tl})
                refined = [e for e in find_events(list(merged.values()), cfg)
                           if lo - step <= e["t"] <= hi + step]
                e = refined[0] if refined else ev
                events.append(Event(rid, e["t"], rule["description_ru"],
                                    pick_evidence(frames, e["t"], e["peak"]), e["score"]))
        return events


class SeatOccupancyDetector(Detector):
    """Правила на признаке «кассир за столом» от детектора людей (без VLM).

    Считает боксы людей по кадрам, строит признак cashier_present и отдаёт его
    в тот же движок правил. Детерминировано и в разы дешевле VLM.
    """

    name = "seat_occupancy"

    def __init__(self, rules: dict, min_score: float = 0.35, batch: int = 8):
        self.rules = rules
        self.min_score, self.batch = min_score, batch
        self._model = None

    def _load(self):
        import torch
        from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                                  fasterrcnn_mobilenet_v3_large_320_fpn)
        self.torch = torch
        self.device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).eval().to(self.device)

    def detect(self, frames: List[Frame]) -> List[List[List[float]]]:
        if self._model is None:
            self._load()
        import cv2
        torch = self.torch
        out = []
        for i in range(0, len(frames), self.batch):
            chunk = frames[i:i + self.batch]
            xs = [torch.from_numpy(cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                  .float().div(255).to(self.device) for f in chunk]
            with torch.inference_mode():
                res = self._model(xs)
            for r in res:
                out.append([[float(v) for v in b] for b, s, l in zip(r["boxes"], r["scores"], r["labels"])
                            if int(l) == 1 and float(s) >= self.min_score])
        return out

    def run(self, frames: List[Frame]) -> List[Event]:
        if not frames or not self.rules:
            return []
        from .occupancy import timeline
        boxes = self.detect(frames)
        events: List[Event] = []
        for rid, rule in self.rules.items():
            z = rule.get("zone") or {}
            tl = timeline([f.t for f in frames], boxes, z.get("dx", 0.8), z.get("dy", 0.4))
            for e in find_events(tl, rule["engine"]):
                events.append(Event(rid, e["t"], rule["description_ru"],
                                    pick_evidence(frames, e["t"], e["peak"]), e["score"]))
        return events


def build_detectors() -> List[Detector]:
    from .rules import ENGINE_RULES, PREDICATES
    occ = {k: r for k, r in ENGINE_RULES.items() if r.get("detector") == "seat_occupancy"}
    vlm = {k: r for k, r in ENGINE_RULES.items() if r.get("detector") != "seat_occupancy"}
    out: List[Detector] = []
    if occ and os.environ.get("USE_OCCUPANCY", "1") == "1":
        try:
            import torchvision  # noqa: F401  (в CPU-образе его нет — правило просто выключится)
            out.append(SeatOccupancyDetector(occ))
        except ImportError:
            pass
    if vlm and os.environ.get("USE_VLM") == "1":
        from .predicates import VLMPredicateExtractor
        out.append(RuleEngineDetector(VLMPredicateExtractor(PREDICATES), vlm))
    return out or [NullDetector()]
