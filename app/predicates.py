"""Извлечение покадровых предикатов (да/нет) локальной VLM.

Идея: VLM не «рассуждает», а отвечает на узкие вопросы; вероятность берётся из
логитов токенов Yes/No за один прямой проход. Это быстро, калибруется порогами
на ваших видео, а текст описания нарушения собирается шаблоном, а не генерируется.

ВАЖНО: код написан без возможности запуска моделей в моей среде. Проверьте на своём
GPU, начните с MODEL_ID=Qwen/Qwen2-VL-2B-Instruct (проверьте лицензию на карточке).
"""
import os
import time
from typing import Dict, List, Optional


class PredicateExtractor:
    def reset(self):
        pass

    def timeline(self, frames, predicate_ids, deadline: Optional[float] = None) -> List[dict]:
        raise NotImplementedError


class CachedExtractor(PredicateExtractor):
    """Читает заранее посчитанные вероятности (см. eval/cache_predicates.py)."""

    def __init__(self, rows: List[dict]):
        self.rows = {round(r["t"], 2): r["p"] for r in rows}

    def timeline(self, frames, predicate_ids, deadline=None):
        out = []
        for f in frames:
            p = self.rows.get(round(f.t, 2))
            if p is not None:
                out.append({"t": f.t, "p": {k: p[k] for k in predicate_ids if k in p}})
        return out


class VLMPredicateExtractor(PredicateExtractor):
    def __init__(self, spec: dict, model_id: Optional[str] = None,
                 max_pixels: Optional[int] = None, min_pixels: int = 64 * 28 * 28):
        self.spec = spec
        self.model_id = model_id or os.environ.get("MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
        # MAX_TOKENS: сколько «визуальных токенов» на кадр (меньше = быстрее и меньше памяти GPU)
        self.max_pixels = max_pixels or int(os.environ.get("MAX_TOKENS", "320")) * 28 * 28
        self.min_pixels = min_pixels
        self.cache: Dict[float, Dict[str, float]] = {}
        self.model = None

    def reset(self):
        self.cache = {}

    def _load(self):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM
        self.torch = torch
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        if os.environ.get("LOAD_4BIT") == "1" and self.device == "cuda":
            # 4-битная загрузка: ~в 3 раза меньше памяти GPU (нужен pip install bitsandbytes)
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoVLM.from_pretrained(
                self.model_id, quantization_config=q, device_map="cuda", local_files_only=offline).eval()
        else:
            self.model = AutoVLM.from_pretrained(
                self.model_id, torch_dtype=dtype, local_files_only=offline).to(self.device).eval()
        tok = self.processor.tokenizer
        enc = lambda w: tok.encode(w, add_special_tokens=False)[0]
        self.yes_ids = sorted({enc(w) for w in ("Yes", "yes", " Yes", " yes")})
        self.no_ids = sorted({enc(w) for w in ("No", "no", " No", " no")})

    def _pil(self, frame):
        import cv2
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))

    def _p_yes(self, pil, question: str) -> float:
        torch = self.torch
        text_q = f"{self.spec['scene_context']} {question} {self.spec['suffix']}"
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_q}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[pil], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        ly = torch.logsumexp(logits[self.yes_ids], 0)
        ln = torch.logsumexp(logits[self.no_ids], 0)
        return torch.sigmoid(ly - ln).item()

    def timeline(self, frames, predicate_ids, deadline=None):
        if self.model is None:
            self._load()
        out = []
        for f in frames:
            if deadline and time.time() > deadline:
                break
            row = self.cache.setdefault(round(f.t, 2), {})
            pil = None
            for pid in predicate_ids:
                if pid not in row:
                    pil = pil or self._pil(f)
                    row[pid] = self._p_yes(pil, self.spec["predicates"][pid])
            out.append({"t": f.t, "p": {pid: row[pid] for pid in predicate_ids}})
        return out
