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


def build_detectors() -> List[Detector]:
    from .rules import ENGINE_RULES, PREDICATES
    if os.environ.get("USE_VLM") == "1":
        from .predicates import VLMPredicateExtractor
        return [RuleEngineDetector(VLMPredicateExtractor(PREDICATES), ENGINE_RULES)]
    return [NullDetector()]
