"""
EvoWrap Perception Module — multi-modal input fusion & drift detection.

Addresses survey critiques:
  - Most agents are text-only; EvoWrap fuses text, images, and arbitrary
    sensor streams through a unified adapter pattern.
  - Environment changes (distribution drift) are *detected*, not ignored,
    triggering the optimizer to start an evolution cycle.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .utils import (
    Experience,
    cosine_sim,
    get_logger,
    image_embed,
    text_embed,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Modality enum & adapter registry
# ---------------------------------------------------------------------------
class Modality(Enum):
    TEXT = auto()
    IMAGE = auto()
    SENSOR = auto()
    CUSTOM = auto()


@dataclass
class ModalityAdapter:
    """Maps a raw input to a fixed-dim embedding vector."""
    name: str
    modality: Modality
    embed_fn: Callable[[Any], np.ndarray]


# Built-in adapters
_TEXT_ADAPTER = ModalityAdapter("text", Modality.TEXT, text_embed)

def _default_image_embed(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img
    return image_embed(gray)

_IMAGE_ADAPTER = ModalityAdapter("image", Modality.IMAGE, _default_image_embed)

_SENSOR_ADAPTER = ModalityAdapter(
    "sensor",
    Modality.SENSOR,
    lambda data: np.array(data, dtype=np.float32).ravel()[:128]
    / (np.linalg.norm(np.array(data, dtype=np.float32).ravel()[:128]) + 1e-9),
)


# ---------------------------------------------------------------------------
# Drift detector — monitors embedding distribution for concept shift
# ---------------------------------------------------------------------------
class DriftDetector:
    """
    Lightweight Page-Hinkley-style drift detector operating on embedding
    cosine similarity.  When consecutive inputs diverge from the running
    centroid beyond *threshold*, a drift event fires.

    After sustained drift (recovery_window consecutive detections), the
    detector performs a soft reset — recomputing the centroid from the
    current history window so it adapts to the new distribution.

    This addresses RQ6 from the lifelong survey: agents must detect when
    the environment has changed so they can trigger self-evolution rather
    than silently degrading.
    """

    def __init__(
        self,
        window: int = 50,
        threshold: float = 0.35,
        embed_dim: int = 128,
        recovery_window: int = 10,
    ):
        self.window = window
        self.threshold = threshold
        self.recovery_window = recovery_window
        self._history: deque[np.ndarray] = deque(maxlen=window)
        self._centroid = np.zeros(embed_dim, dtype=np.float32)
        self._count = 0
        self._drift_score = 0.0
        self._consecutive_drift = 0

    def update(self, embedding: np.ndarray) -> bool:
        """Feed a new embedding; returns True if drift is detected."""
        self._count += 1
        self._history.append(embedding)

        # Update running centroid (exponential moving average)
        alpha = min(1.0, 2.0 / (self._count + 1))
        self._centroid = (1 - alpha) * self._centroid + alpha * embedding

        if len(self._history) < 5:
            return False

        # Compute mean similarity of recent window to centroid
        sims = [cosine_sim(e, self._centroid) for e in self._history]
        mean_sim = float(np.mean(sims))
        self._drift_score = 1.0 - mean_sim

        drifted = self._drift_score > self.threshold
        if drifted:
            self._consecutive_drift += 1
            logger.info(
                "Drift detected (score=%.3f > threshold=%.3f, consecutive=%d)",
                self._drift_score,
                self.threshold,
                self._consecutive_drift,
            )
            # Soft reset after sustained drift to adapt to new distribution
            if self._consecutive_drift >= self.recovery_window:
                self._soft_reset()
        else:
            self._consecutive_drift = 0

        return drifted

    def _soft_reset(self) -> None:
        """
        Recompute centroid from current history window and reset count
        so alpha becomes large again (~0.1-0.2), allowing fast adaptation
        to the new distribution.
        """
        if self._history:
            self._centroid = np.mean(
                np.stack(list(self._history)), axis=0
            ).astype(np.float32)
        self._count = len(self._history)
        self._consecutive_drift = 0
        logger.info(
            "Drift soft-reset: centroid recomputed from %d samples, "
            "alpha will be ~%.3f",
            len(self._history),
            min(1.0, 2.0 / (self._count + 1)),
        )

    @property
    def drift_score(self) -> float:
        return self._drift_score

    def reset(self) -> None:
        self._history.clear()
        self._centroid[:] = 0.0
        self._count = 0
        self._drift_score = 0.0
        self._consecutive_drift = 0


# ---------------------------------------------------------------------------
# Perception module — the public API
# ---------------------------------------------------------------------------
class PerceptionModule:
    """
    Fuses multi-modal inputs into a single embedding vector and watches
    for distribution drift.

    Usage:
        pm = PerceptionModule()
        pm.register_adapter(my_lidar_adapter)
        embedding, drifted = pm.perceive({"text": "hello", "image": img_arr})
    """

    def __init__(
        self,
        embed_dim: int = 128,
        drift_window: int = 50,
        drift_threshold: float = 0.35,
        drift_recovery_window: int = 10,
    ):
        self.embed_dim = embed_dim
        self._adapters: Dict[str, ModalityAdapter] = {
            "text": _TEXT_ADAPTER,
            "image": _IMAGE_ADAPTER,
            "sensor": _SENSOR_ADAPTER,
        }
        self._drift = DriftDetector(
            drift_window, drift_threshold, embed_dim, drift_recovery_window,
        )

    # -- adapter management --------------------------------------------------
    def register_adapter(self, adapter: ModalityAdapter) -> None:
        self._adapters[adapter.name] = adapter
        logger.info("Registered adapter '%s' (%s)", adapter.name, adapter.modality.name)

    # -- core perceive -------------------------------------------------------
    def perceive(self, inputs: Dict[str, Any]) -> tuple[np.ndarray, bool]:
        """
        Process a dict of {modality_name: raw_data} → (fused_embedding, drift_flag).

        Fusion strategy: mean-pool normalized per-modality embeddings.
        """
        embeddings: List[np.ndarray] = []

        for key, data in inputs.items():
            adapter = self._adapters.get(key)
            if adapter is None:
                logger.warning("No adapter for modality '%s' — skipping", key)
                continue
            try:
                vec = adapter.embed_fn(data)
                # Pad or truncate to embed_dim
                if vec.size < self.embed_dim:
                    vec = np.pad(vec, (0, self.embed_dim - vec.size))
                elif vec.size > self.embed_dim:
                    vec = vec[: self.embed_dim]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                embeddings.append(vec.astype(np.float32))
            except Exception as exc:
                logger.error("Adapter '%s' failed: %s", key, exc)

        if not embeddings:
            fused = np.zeros(self.embed_dim, dtype=np.float32)
        else:
            fused = np.mean(embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(fused)
            if norm > 0:
                fused /= norm

        drifted = self._drift.update(fused)
        return fused, drifted

    def reset_drift(self) -> None:
        self._drift.reset()

    @property
    def drift_score(self) -> float:
        return self._drift.drift_score
