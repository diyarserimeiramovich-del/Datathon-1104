"""python -m eval.labels open_labels.xlsx > labels.csv
В xlsx имена смешивают латинскую X и кириллическую Х, поэтому имя нормализуем."""
import csv
import re
import sys

import openpyxl


def norm(name: str) -> str:
    name = str(name).strip().replace("Х", "X").replace("х", "x")
    return re.sub(r"_\d+$", "", name)  # X-1_1 -> X-1 (суффикс добавляет загрузчик)


def parse_ts(s):
    s = str(s or "")
    m = re.search(r"(\d+)\s*минут", s)
    sec = re.search(r"(\d+)\s*секунд", s)
    if not sec:
        return ""
    return (int(m.group(1)) if m else 0) * 60 + int(sec.group(1))


def main(path):
    ws = openpyxl.load_workbook(path, read_only=True).active
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["video", "label", "timecode"])
    for name, ts, lab in ws.iter_rows(min_row=2, values_only=True):
        if name:
            w.writerow([norm(name), "no_violation" if "без" in str(lab) else "violation", parse_ts(ts)])


if __name__ == "__main__":
    main(sys.argv[1])
