import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
_RULES = json.loads((ROOT / "rules.json").read_text(encoding="utf-8"))
RULES = {r["id"]: r for r in _RULES["violation_types"]}
ENGINE_RULES = {k: r for k, r in RULES.items() if r.get("enabled") and r.get("engine")}
PREDICATES = json.loads((ROOT / "predicates.json").read_text(encoding="utf-8"))


def rule_text(rule_id: str) -> str:
    """Ссылка на норму. Непроверенные ссылки честно помечаются как неподтверждённые."""
    r = RULES.get(rule_id)
    if not r:
        return "Источник не определён"
    s = r["source"]
    if s.get("status") != "VERIFIED" or not s.get("document"):
        return "Нормативный источник уточняется (ссылка не подтверждена)"
    art = f", {s['article']}" if s.get("article") else ""
    return f"{s['document']}{art}"
