import base64
from dataclasses import dataclass

import cv2


@dataclass
class Frame:
    t: float          # секунды от начала
    image: "cv2.Mat"  # BGR


def read_meta(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return fps, n / fps if fps else 0.0


def sample_frames(path: str, step_sec: float = 0.5, max_side: int = 960):
    """Равномерная выборка кадров. Для .mov OpenCV обычно справляется;
    если нет, предварительно конвертируйте через ffmpeg в mp4."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * step_sec)))
    i = 0
    out = []
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, img = cap.retrieve()
            if ok:
                h, w = img.shape[:2]
                k = max_side / max(h, w)
                if k < 1:
                    img = cv2.resize(img, (int(w * k), int(h * k)))
                out.append(Frame(t=i / fps, image=img))
        i += 1
    cap.release()
    return out


def to_b64_jpeg(img, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()
