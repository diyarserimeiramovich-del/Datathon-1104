import glob
import json

keys = ["cash_on_desk", "client_seated", "visitor_present", "cashier_away", "phone_visible", "person_on_cashier_side"]
with open("out.txt", "w", encoding="utf-8") as out:
    out.write("t  cash client visitor away phone side\n")
    for f in sorted(glob.glob("cache/*.json")):
        d = json.load(open(f, encoding="utf-8"))
        out.write("== " + d["video"] + "\n")
        for r in d["timeline"]:
            p = r["p"]
            out.write(f"{r['t']:5.1f} " + " ".join(f"{int(p[k] * 100):3d}" for k in keys) + "\n")
print("готово, откройте out.txt")
