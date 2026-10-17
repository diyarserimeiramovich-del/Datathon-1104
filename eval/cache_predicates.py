"""python -m eval.cache_predicates videos_dir cache_dir
Один раз считает вероятности ВСЕХ предикатов на сетке 0.5 с и сохраняет в JSON.
Дальше пороги в rules.json меняются и проверяются за секунды (eval/evaluate.py)."""
import json
import os
import sys
from pathlib import Path

from app.predicates import VLMPredicateExtractor
from app.rules import PREDICATES
from app.video import sample_frames
from eval.labels import norm


def main(vdir, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    ex = VLMPredicateExtractor(PREDICATES)
    ids = sorted(PREDICATES["predicates"])
    for path in sorted(Path(vdir).glob("*.mov")):
        name = norm(path.stem)
        dst = Path(out) / f"{name}.json"
        if dst.exists():
            continue
        frames = sample_frames(str(path), step_sec=float(os.environ.get("STEP_SEC", "0.5")))
        ex.reset()
        tl = ex.timeline(frames, ids)
        dst.write_text(json.dumps({"video": name, "timeline": tl}), encoding="utf-8")
        print("cached", name, len(tl), "frames")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
