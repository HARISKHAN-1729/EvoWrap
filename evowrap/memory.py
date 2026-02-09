
from __future__ import annotations

import json
import math
import os
import pickle
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import Experience, RingBuffer, cosine_sim, get_logger, text_embed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Numpy-based vector index (brute-force cosine — works up to ~100k vectors)
# ---------------------------------------------------------------------------
class _NumpyIndex:
    """
    Minimal in-process vector store using brute-force cosine similarity.
    Fallback when FAISS is not installed.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self._vecs: List[np.ndarray] = []
        self._payloads: List[Any] = []

    def add(self, vector: np.ndarray, payload: Any = None) -> int:
        idx = len(self._vecs)
        v = np.asarray(vector, dtype=np.float32)
        if v.size != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {v.size}")
        self._vecs.append(v)
        self._payloads.append(payload)
        return idx

    def query(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, Any]]:
        """Return [(similarity, payload), ...] sorted descending."""
        if not self._vecs:
            return []
        q = np.asarray(vector, dtype=np.float32)
        mat = np.stack(self._vecs)                     # (N, dim)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat_n = mat / norms
        q_n = q / (np.linalg.norm(q) + 1e-9)
        sims = (mat_n @ q_n).ravel()
        top_idx = np.argsort(-sims)[: top_k]
        return [(float(sims[i]), self._payloads[i]) for i in top_idx]

    def __len__(self) -> int:
        return len(self._vecs)

    def save(self, path: str) -> None:
        """Save index to disk using pickle."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "dim": self.dim,
            "vecs": self._vecs,
            "payloads": self._payloads,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.debug("NumpyIndex saved to %s (%d vectors)", path, len(self))

    @classmethod
    def load(cls, path: str) -> "_NumpyIndex":
        """Load index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(dim=data["dim"])
        idx._vecs = data["vecs"]
        idx._payloads = data["payloads"]
        logger.debug("NumpyIndex loaded from %s (%d vectors)", path, len(idx))
        return idx


# ---------------------------------------------------------------------------
# FAISS-based vector index (efficient similarity search)
# ---------------------------------------------------------------------------
class FaissIndex:
    """
    FAISS-backed vector index using inner-product search on normalised vectors
    (equivalent to cosine similarity). Requires faiss-cpu to be installed.
    """

    def __init__(self, dim: int = 128, index_type: str = "flat"):
        import faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs
        self._payloads: List[Any] = []

    def add(self, vector: np.ndarray, payload: Any = None) -> int:
        idx = len(self._payloads)
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if v.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {v.shape[1]}")
        # Normalize for cosine similarity via inner product
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        self._index.add(v)
        self._payloads.append(payload)
        return idx

    def query(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, Any]]:
        """Return [(similarity, payload), ...] sorted descending."""
        if not self._payloads:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        k = min(top_k, len(self._payloads))
        scores, indices = self._index.search(q, k)
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            if idx >= 0:
                results.append((float(scores[0][i]), self._payloads[idx]))
        return results

    def __len__(self) -> int:
        return len(self._payloads)

    def save(self, path: str) -> None:
        """Save FAISS index + payloads to disk."""
        import faiss
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        faiss.write_index(self._index, path + ".faiss")
        with open(path + ".payloads", "wb") as f:
            pickle.dump({"dim": self.dim, "payloads": self._payloads}, f)
        logger.debug("FaissIndex saved to %s (%d vectors)", path, len(self))

    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        """Load FAISS index + payloads from disk."""
        import faiss
        idx = cls.__new__(cls)
        idx._index = faiss.read_index(path + ".faiss")
        with open(path + ".payloads", "rb") as f:
            data = pickle.load(f)
        idx.dim = data["dim"]
        idx._payloads = data["payloads"]
        logger.debug("FaissIndex loaded from %s (%d vectors)", path, len(idx))
        return idx


# ---------------------------------------------------------------------------
# Factory: pick best available backend
# ---------------------------------------------------------------------------
def make_vector_index(dim: int = 128, backend: str = "auto"):
    """
    Create a vector index with the best available backend.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    backend : str
        "auto" (try faiss, fall back to numpy), "faiss", or "numpy".
    """
    if backend == "numpy":
        return _NumpyIndex(dim)

    if backend in ("auto", "faiss"):
        try:
            import faiss  # noqa: F401
            logger.info("Using FAISS backend for vector index (dim=%d)", dim)
            return FaissIndex(dim)
        except ImportError:
            if backend == "faiss":
                raise ImportError("faiss-cpu is required for backend='faiss'. Install with: pip install faiss-cpu")
            logger.info("FAISS not available, falling back to numpy backend (dim=%d)", dim)
            return _NumpyIndex(dim)

    raise ValueError(f"Unknown backend: {backend!r}. Use 'auto', 'faiss', or 'numpy'.")


# Backward-compatible alias
VectorIndex = _NumpyIndex


# ---------------------------------------------------------------------------
# Importance tracker (EWC-style)
# ---------------------------------------------------------------------------
class ImportanceTracker:
    """
    Tracks per-experience importance via accumulated reward magnitude.
    The optimizer can query importance weights to penalise updates that
    would damage high-importance memories — mitigating catastrophic forgetting
    (stability-plasticity dilemma, lifelong survey Section 1).
    """

    def __init__(self, decay: float = 0.99):
        self._scores: Dict[str, float] = {}  # uid → importance
        self._decay = decay

    def update(self, uid: str, reward: float) -> None:
        prev = self._scores.get(uid, 0.0)
        self._scores[uid] = self._decay * prev + abs(reward)

    def importance(self, uid: str) -> float:
        return self._scores.get(uid, 0.0)

    def top_k(self, k: int = 20) -> List[Tuple[str, float]]:
        return sorted(self._scores.items(), key=lambda x: -x[1])[:k]

    def regularization_penalty(self, uids: Sequence[str]) -> float:
        """Return a scalar penalty proportional to total importance of given uids."""
        return sum(self.importance(u) for u in uids)


# ---------------------------------------------------------------------------
# Memory Module — public API
# ---------------------------------------------------------------------------
class MemoryModule:
    """
    Hybrid memory combining:
      1. **Short-term buffer** — recent experiences in a ring buffer for fast
         replay during the current episode.
      2. **Long-term vector store** — all experiences indexed by embedding for
         similarity retrieval across the agent's entire lifetime.
      3. **Importance tracker** — EWC-inspired weights to protect critical memories
         from being overwritten.

    Together, these give the wrapped agent a "hippocampus + neocortex" architecture
    that no existing wrapper (Mem0, Letta, Arc) provides in one package.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        short_term_capacity: int = 1_000,
        consolidation_interval: int = 100,
        vector_backend: str = "auto",
    ):
        self.embed_dim = embed_dim
        self.short_term = RingBuffer(short_term_capacity)
        self.long_term = make_vector_index(embed_dim, backend=vector_backend)
        self.importance = ImportanceTracker()
        self._consolidation_interval = consolidation_interval
        self._step = 0

        # Task-partitioned episodic memory (maps task_id → list of uids)
        self._episodes: Dict[str, List[str]] = defaultdict(list)

    # -- store ---------------------------------------------------------------
    def store(
        self,
        experience: Experience,
        embedding: Optional[np.ndarray] = None,
        task_id: str = "default",
    ) -> None:
        """Store an experience in both short- and long-term memory."""
        if embedding is not None:
            experience.embedding = embedding

        if experience.embedding is None:
            # Fallback: embed the string representation
            experience.embedding = text_embed(str(experience.state))

        self.short_term.append(experience)
        self.long_term.add(experience.embedding, experience)
        self.importance.update(experience.uid, experience.reward)
        self._episodes[task_id].append(experience.uid)

        self._step += 1
        if self._step % self._consolidation_interval == 0:
            self._consolidate()

    # -- retrieval -----------------------------------------------------------
    def recall(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Experience]:
        """Retrieve the most relevant long-term experiences."""
        results = self.long_term.query(query_embedding, top_k)
        return [exp for _, exp in results]

    def replay(self, batch_size: int = 32) -> List[Experience]:
        """Sample a random batch from short-term memory (experience replay)."""
        return self.short_term.sample(batch_size)

    def replay_for_task(self, task_id: str, k: int = 10) -> List[Experience]:
        """Replay experiences from a specific past task to fight forgetting."""
        uids = self._episodes.get(task_id, [])
        if not uids:
            return []
        # Retrieve by uid from long-term (linear scan — acceptable at scale)
        all_exps = [p for _, p in self.long_term.query(
            np.zeros(self.embed_dim), top_k=len(self.long_term)
        )]
        task_exps = [e for e in all_exps if e.uid in set(uids)]
        return random.sample(task_exps, min(k, len(task_exps)))

    # -- regularization signal for optimizer ---------------------------------
    def forgetting_penalty(self, task_id: str = "default") -> float:
        """
        Scalar penalty reflecting how important the memories for *task_id* are.
        The optimizer adds this to its loss to avoid catastrophic forgetting.
        """
        uids = self._episodes.get(task_id, [])
        return self.importance.regularization_penalty(uids)

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save full memory state to a directory.

        Creates:
          path/long_term.*      — vector index files
          path/state.pkl        — short-term buffer, importance scores, episodes, step
        """
        os.makedirs(path, exist_ok=True)
        # Save long-term index
        lt_path = os.path.join(path, "long_term")
        self.long_term.save(lt_path)

        # Save remaining state
        state = {
            "embed_dim": self.embed_dim,
            "short_term": list(self.short_term),
            "importance_scores": self.importance._scores,
            "episodes": dict(self._episodes),
            "step": self._step,
            "consolidation_interval": self._consolidation_interval,
            "backend_type": type(self.long_term).__name__,
        }
        with open(os.path.join(path, "state.pkl"), "wb") as f:
            pickle.dump(state, f)
        logger.info("MemoryModule saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "MemoryModule":
        """Load full memory state from a directory."""
        with open(os.path.join(path, "state.pkl"), "rb") as f:
            state = pickle.load(f)

        mm = cls(
            embed_dim=state["embed_dim"],
            consolidation_interval=state["consolidation_interval"],
        )

        # Load long-term index
        lt_path = os.path.join(path, "long_term")
        backend_type = state.get("backend_type", "_NumpyIndex")
        if backend_type == "FaissIndex":
            try:
                mm.long_term = FaissIndex.load(lt_path)
            except ImportError:
                logger.warning("FAISS not available, loading as NumpyIndex")
                mm.long_term = _NumpyIndex.load(lt_path)
        else:
            mm.long_term = _NumpyIndex.load(lt_path)

        # Restore state
        for exp in state["short_term"]:
            mm.short_term.append(exp)
        mm.importance._scores = state["importance_scores"]
        mm._episodes = defaultdict(list, state["episodes"])
        mm._step = state["step"]
        logger.info("MemoryModule loaded from %s", path)
        return mm

    # -- stats ---------------------------------------------------------------
    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "short_term_size": len(self.short_term),
            "long_term_size": len(self.long_term),
            "tasks_tracked": list(self._episodes.keys()),
            "total_steps": self._step,
        }

    # -- internal ------------------------------------------------------------
    def _consolidate(self) -> None:
        """
        Periodic consolidation: log stats (placeholder for more advanced
        strategies like memory pruning or summarisation).
        """
        logger.debug(
            "Memory consolidation @ step %d | ST=%d LT=%d",
            self._step,
            len(self.short_term),
            len(self.long_term),
        )
