"""python -m eval.detect_boxes videos_dir cache_boxes_dir [step_sec]
Детектор людей (torchvision FasterRCNN-MobileNet) по кадрам, только боксы — без VLM.
Боксы кэшируются, чтобы признаки «кассир за столом» можно было перебирать без GPU."""
import json
import os
import sys
from pathlib import Path

from app.video import sample_frames
from eval.labels import norm


def load_model():
    import torch
    from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                              fasterrcnn_mobilenet_v3_large_320_fpn)
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    m = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).eval().to(device)
    return m, device


def run(model, device, frames, batch=8, score_min=0.35):
    import cv2
    import torch
    out = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        xs = [torch.from_numpy(cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
              .float().div(255).to(device) for f in chunk]
        with torch.inference_mode():
            res = model(xs)
        for f, r in zip(chunk, res):
            boxes = [[round(float(v), 1) for v in b] + [round(float(s), 3)]
                     for b, s, l in zip(r["boxes"], r["scores"], r["labels"])
                     if int(l) == 1 and float(s) >= score_min]
            out.append({"t": round(f.t, 2), "boxes": boxes})
    return out


def main(vdir, out, step=0.5):
    Path(out).mkdir(parents=True, exist_ok=True)
    model, device = load_model()
    files = {norm(p.stem): p for p in Path(vdir).iterdir()
             if p.suffix.lower() in (".mov", ".mp4")}
    for name in sorted(files):
        dst = Path(out) / f"{name}.json"
        if dst.exists():
            continue
        frames = sample_frames(str(files[name]), step_sec=float(step))
        rows = run(model, device, frames)
        dst.write_text(json.dumps({"video": name, "step": float(step), "frames": rows}),
                       encoding="utf-8")
        print("boxes", name, len(rows), "кадров", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2] if len(a) > 2 else 0.5)
