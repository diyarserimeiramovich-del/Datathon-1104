import json
import os
import secrets
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from .pipeline import analyze

API_KEY = os.environ.get("API_KEY", "")
DATA = Path(os.environ.get("DATA_DIR", "data"))
(DATA / "jobs").mkdir(parents=True, exist_ok=True)
(DATA / "uploads").mkdir(parents=True, exist_ok=True)
MAX_BYTES = 300 * 1024 * 1024

app = FastAPI(title="Cashier discipline AI")
pool = ThreadPoolExecutor(max_workers=1)  # одна GPU, обрабатываем по очереди


def auth(x_api_key: str = Header(default="")):
    if not API_KEY or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _save(job_id: str, obj: dict):
    tmp = DATA / "jobs" / f"{job_id}.tmp"
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(DATA / "jobs" / f"{job_id}.json")


def _run(job_id: str, path: Path):
    try:
        res = analyze(str(path))
        _save(job_id, {"job_id": job_id, "status": "done", **res})
    except Exception as e:  # noqa: BLE001
        _save(job_id, {"job_id": job_id, "status": "failed", "error": str(e)})
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/v1/analyze", dependencies=[Depends(auth)])
async def post_analyze(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not name.endswith((".mov", ".mp4")):
        raise HTTPException(400, "Ожидается видеофайл .mov")
    job_id = uuid.uuid4().hex[:8]
    path = DATA / "uploads" / f"{job_id}.mov"
    size = 0
    with path.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                f.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, "Файл слишком большой")
            f.write(chunk)
    _save(job_id, {"job_id": job_id, "status": "processing"})
    pool.submit(_run, job_id, path)
    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(400, "bad id")
    p = DATA / "jobs" / f"{job_id}.json"
    if not p.exists():
        raise HTTPException(404, "job not found")
    return json.loads(p.read_text(encoding="utf-8"))


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
