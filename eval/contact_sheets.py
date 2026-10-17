"""python -m eval.contact_sheets labels.csv videos_dir [out_dir]
Нарезает по метке контактный лист: 6 кадров вокруг таймкода с подписями секунд.
Нужен, чтобы проставить тип нарушения в data/labels_types.csv, не пересматривая видео целиком.
Контактные листы не попадают в git (data/contacts/ в .gitignore)."""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from eval.labels import norm

OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
COLS, TILE_W = 4, 480


def frames_at(cap, times):
    """Один последовательный проход: seek (CAP_PROP_POS_MSEC) ломается на части .mov."""
    targets = sorted({round(max(0.0, g + o), 2) for g in times for o in OFFSETS})
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    got, i, ti = {}, 0, 0
    while ti < len(targets):
        if not cap.grab():
            break
        t = i / fps
        while ti < len(targets) and t >= targets[ti] - 0.5 / fps:
            ok, img = cap.retrieve()
            got[targets[ti]] = img if ok else None
            ti += 1
        i += 1
    return got


def sheet(path, times, dst):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"не открыть {path}")
    got = frames_at(cap, times)
    tiles = []
    for g in times:
        for off in OFFSETS:
            img = got.get(round(max(0.0, g + off), 2))
            if img is None:
                continue
            k = TILE_W / img.shape[1]
            img = cv2.resize(img, (TILE_W, int(img.shape[0] * k)))
            label = f"{g + off:+.0f}s" + ("  <= METKA" if off == 0 else "")
            cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(img, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (0, 255, 255) if off == 0 else (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(img)
    cap.release()
    if not tiles:
        return 0
    h = max(t.shape[0] for t in tiles)
    blank = np.full((h, TILE_W, 3), 32, np.uint8)
    rows = []
    for i in range(0, len(tiles), COLS):
        r = tiles[i:i + COLS]
        r += [blank] * (COLS - len(r))
        rows.append(cv2.hconcat([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                                    cv2.BORDER_CONSTANT, value=(32, 32, 32)) for t in r]))
    mat = cv2.vconcat(rows)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), mat, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return len(tiles)


def main(labels, vdir, out="data/contacts", types_csv="data/labels_types.csv"):
    rows = [r for r in csv.DictReader(open(labels, encoding="utf-8")) if r["label"] == "violation"]
    # имена в видео могут быть с кириллической Х, в метках — с латинской
    files = {norm(p.stem): p for p in Path(vdir).iterdir() if p.suffix.lower() in (".mov", ".mp4")}
    seen, made = set(), 0
    for r in rows:
        name = r["video"]
        if name in seen:      # дубликаты (X-47 = X-4, X-48 = X-5) не режем дважды
            continue
        seen.add(name)
        path = files.get(name)
        if path is None:
            print("нет видео:", name)
            continue
        times = [float(x) for x in (r.get("timecodes_sec") or r.get("timecode") or "").split(";") if x.strip()]
        n = sheet(path, times, Path(out) / f"{name}.jpg")
        print(f"{name}: {n} кадров -> {out}/{name}.jpg")
        made += 1
    t = Path(types_csv)
    if not t.exists():
        with open(t, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video", "timecode", "type", "comment"])
            for r in rows:
                w.writerow([r["video"], r.get("timecodes_sec") or r.get("timecode"), "", ""])
        print("шаблон создан:", t)
    gallery = Path(out) / "index.html"
    names = sorted(seen, key=lambda n: int(n.split("-")[1]))
    cards = "\n".join(f'<h3>{n}</h3><img src="{n}.jpg" loading="lazy">' for n in names)
    gallery.write_text(
        "<!doctype html><meta charset=utf-8><title>Контактные листы</title>"
        "<style>body{background:#111;color:#ddd;font:14px system-ui;margin:16px}"
        "img{width:100%;max-width:1400px;display:block;margin:6px 0 18px}</style>"
        f"<h1>Контактные листы ({len(names)})</h1>"
        "<p>Жёлтая подпись — секунды от начала клипа, кадр метки помечен &lt;= METKA. "
        "Тип нарушения вписывайте в data/labels_types.csv (см. docs/05_LABEL_TYPES.md).</p>"
        + cards, encoding="utf-8")
    print(f"листов: {made}, галерея: {gallery}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*a)
