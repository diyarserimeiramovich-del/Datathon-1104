import time

from .detectors import build_detectors
from .rules import RULES, rule_text
from .video import sample_frames, to_b64_jpeg

TIME_BUDGET_SEC = 240  # лимит жюри: 5 минут, оставляем запас
_detectors = None


def analyze(path: str) -> dict:
    global _detectors
    if _detectors is None:
        _detectors = build_detectors()  # модели грузятся один раз
    t0 = time.time()
    frames = sample_frames(path)
    events = []
    for d in _detectors:
        if time.time() - t0 > TIME_BUDGET_SEC:
            break
        events.extend(d.run(frames))

    events.sort(key=lambda e: e.t)
    violations = []
    for e in events:
        rule = RULES.get(e.rule_id, {})
        violations.append({
            "type": rule.get("type", e.rule_id),
            "timecode_sec": round(e.t, 1),
            "description": e.description,
            "source": rule_text(e.rule_id),
            "evidence_frames": [
                {"timestamp_sec": round(f.t, 1), "image_base64": to_b64_jpeg(f.image)}
                for f in e.frames[:3]
            ],
        })
    return {"verdict": "violation" if violations else "no_violation",
            "violations": violations}
