
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FMT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": record.created,
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        })


def enable_json_logging(path: Optional[str] = None) -> logging.Handler:
    """
    Add a JSON file handler to all evowrap loggers.

    Parameters
    ----------
    path : str, optional
        Output file path. Defaults to ``evowrap_logs.jsonl``.

    Returns
    -------
    logging.Handler
        The file handler that was added (useful for cleanup in tests).
    """
    if path is None:
        path = "evowrap_logs.jsonl"
    handler = logging.FileHandler(path, mode="a")
    handler.setFormatter(JsonFormatter())
    # Attach to evowrap root logger so all child loggers inherit it
    root = logging.getLogger("evowrap")
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    return handler


# ---------------------------------------------------------------------------
# Metrics exporter (structured JSONL)
# ---------------------------------------------------------------------------
class MetricsExporter:
    """
    Append-only JSONL metrics file for observability.

    Usage:
        exporter = MetricsExporter("metrics.jsonl")
        exporter.export({"reward": 0.95, "step": 100})
        all_metrics = exporter.read()
    """

    def __init__(self, path: str = "evowrap_metrics.jsonl"):
        self.path = path

    def export(self, stats: dict) -> None:
        """Append a single metrics entry as a JSON line."""
        entry = {"_timestamp": time.time(), **stats}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def read(self) -> List[dict]:
        """Read all metrics entries."""
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries


# ---------------------------------------------------------------------------
# Lightweight text embedding (bag-of-character-trigrams → fixed-dim vector)
# Good enough for similarity/drift detection without a neural model.
# ---------------------------------------------------------------------------
_EMBED_DIM = 128

def text_embed(text: str, dim: int = _EMBED_DIM) -> np.ndarray:
    """Deterministic, fast text → vector via hashed char-trigram counts."""
    vec = np.zeros(dim, dtype=np.float32)
    t = text.lower().strip()
    for i in range(len(t) - 2):
        trigram = t[i : i + 3]
        idx = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def image_embed(pixels: np.ndarray, dim: int = _EMBED_DIM) -> np.ndarray:
    """Cheap image embedding via spatial average-pooling + PCA-like projection.

    *pixels* can be any shape (H, W) or (H, W, C).  We flatten, downsample,
    then hash-project to *dim* dimensions.
    """
    flat = pixels.astype(np.float32).ravel()
    # Downsample to max 1024 values for speed
    if flat.size > 1024:
        indices = np.linspace(0, flat.size - 1, 1024).astype(int)
        flat = flat[indices]
    # Random-but-deterministic projection
    rng = np.random.RandomState(42)
    proj = rng.randn(flat.size, dim).astype(np.float32) / math.sqrt(flat.size)
    vec = flat @ proj
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Experience tuple — the universal currency of the memory system
# ---------------------------------------------------------------------------
@dataclass
class Experience:
    state: Any                          # raw observation
    action: Any                         # what the agent did
    reward: float = 0.0                 # scalar feedback
    next_state: Any = None
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        raw = f"{self.state}|{self.action}|{self.reward}|{self.timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Ring buffer for bounded collections
# ---------------------------------------------------------------------------
class RingBuffer:
    """Fixed-capacity FIFO with random-sample support (for experience replay)."""

    def __init__(self, capacity: int = 10_000):
        self._buf: deque = deque(maxlen=capacity)

    def append(self, item: Any) -> None:
        self._buf.append(item)

    def sample(self, k: int) -> List[Any]:
        k = min(k, len(self._buf))
        return random.sample(list(self._buf), k)

    def __len__(self) -> int:
        return len(self._buf)

    def __iter__(self):
        return iter(self._buf)


# ---------------------------------------------------------------------------
# Simple timer context manager
# ---------------------------------------------------------------------------
class Timer:
    def __init__(self, label: str = "", logger: Optional[logging.Logger] = None):
        self.label = label
        self.logger = logger
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        if self.logger:
            self.logger.debug("%s took %.4fs", self.label, self.elapsed)
