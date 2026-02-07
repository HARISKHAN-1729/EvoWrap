"""
EvoWrap Safety Module — Three Laws enforcement & ethical guardrails.

Addresses survey critiques:
  - Self-evolving survey's "Three Laws of AI Agent" (Endure, Excel, Evolve)
    are stated as principles but no existing framework *enforces* them at
    runtime.  Agent Lightning optimises performance while ignoring ethical
    risk; Mem0/ToolLLM have no safety layer at all.
  - EvoWrap makes the Three Laws *executable*: every proposed evolution is
    validated through a hierarchical gate (Endure → Excel → Evolve) before
    it can be applied.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Safety verdicts
# ---------------------------------------------------------------------------
class Verdict(Enum):
    PASS = auto()
    FAIL_ENDURE = auto()   # would crash or destabilise
    FAIL_EXCEL = auto()    # would degrade performance
    FAIL_EVOLVE = auto()   # no measurable improvement
    BLOCKED = auto()       # custom guardrail triggered


@dataclass
class SafetyReport:
    verdict: Verdict
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS


# ---------------------------------------------------------------------------
# Guardrails (user-configurable blocked patterns)
# ---------------------------------------------------------------------------
@dataclass
class Guardrail:
    """A named predicate that blocks evolutions matching unwanted patterns."""
    name: str
    check_fn: Callable[[Dict[str, Any]], bool]  # True = violation
    message: str = "Guardrail triggered"


# ---------------------------------------------------------------------------
# Safety Checker — the public API
# ---------------------------------------------------------------------------
class SafetyChecker:
    """
    Enforces the Three Laws on every proposed evolution:

        Law I  — **Endure** (stability):  The proposed change must not crash
                  the agent or cause error-rate spikes.  Checked via dry-run
                  simulation on a sample of recent inputs.
        Law II — **Excel** (performance):  Post-evolution metrics must not
                  drop by more than *perf_tolerance* relative to baseline.
        Law III — **Evolve** (progress):  The change must yield a measurable
                   improvement on at least one target metric.

    The laws are evaluated in order; a failure at any level rejects the
    proposed evolution.  Custom guardrails can add domain-specific blocks
    (e.g., "never delete the safety module itself").
    """

    def __init__(
        self,
        perf_tolerance: float = 0.05,    # max allowed relative performance drop
        min_improvement: float = 0.01,   # min required relative improvement
        crash_sim_trials: int = 10,      # number of dry-run inputs for Law I
        max_error_rate: float = 0.20,    # max fraction of crashes in dry-run
    ):
        self.perf_tolerance = perf_tolerance
        self.min_improvement = min_improvement
        self.crash_sim_trials = crash_sim_trials
        self.max_error_rate = max_error_rate
        self._guardrails: List[Guardrail] = []

    def add_guardrail(self, guardrail: Guardrail) -> None:
        self._guardrails.append(guardrail)
        logger.info("Guardrail added: %s", guardrail.name)

    # -- main validation entry point ----------------------------------------
    def validate(
        self,
        *,
        proposed_agent_fn: Callable[[Any], Any],
        baseline_agent_fn: Callable[[Any], Any],
        test_inputs: List[Any],
        reward_fn: Callable[[Any, Any], float],
        evolution_metadata: Optional[Dict[str, Any]] = None,
    ) -> SafetyReport:
        """
        Run the full Three-Laws check on a proposed evolution.

        Parameters
        ----------
        proposed_agent_fn : callable
            The agent *after* the proposed change.
        baseline_agent_fn : callable
            The agent *before* the change (snapshot).
        test_inputs : list
            Representative inputs for simulation.
        reward_fn : callable
            (input, output) → float scoring function.
        evolution_metadata : dict, optional
            Extra info the guardrails can inspect.
        """
        meta = evolution_metadata or {}

        # --- Custom guardrails (fast reject) --------------------------------
        for g in self._guardrails:
            if g.check_fn(meta):
                return SafetyReport(
                    verdict=Verdict.BLOCKED,
                    details={"guardrail": g.name},
                    suggestions=[g.message],
                )

        # --- Law I: Endure (stability) --------------------------------------
        crash_count = 0
        sample = test_inputs[: self.crash_sim_trials]
        for inp in sample:
            try:
                proposed_agent_fn(inp)
            except Exception:
                crash_count += 1

        error_rate = crash_count / max(len(sample), 1)
        if error_rate > self.max_error_rate:
            return SafetyReport(
                verdict=Verdict.FAIL_ENDURE,
                details={"error_rate": error_rate, "threshold": self.max_error_rate},
                suggestions=[
                    f"Proposed change crashes on {error_rate:.0%} of inputs "
                    f"(limit: {self.max_error_rate:.0%}). Stabilise first."
                ],
            )

        # --- Law II: Excel (no regression) ----------------------------------
        baseline_rewards = self._evaluate(baseline_agent_fn, test_inputs, reward_fn)
        proposed_rewards = self._evaluate(proposed_agent_fn, test_inputs, reward_fn)

        baseline_mean = float(np.mean(baseline_rewards)) if baseline_rewards else 0.0
        proposed_mean = float(np.mean(proposed_rewards)) if proposed_rewards else 0.0

        if baseline_mean > 0:
            relative_drop = (baseline_mean - proposed_mean) / baseline_mean
        else:
            relative_drop = 0.0 if proposed_mean >= baseline_mean else 1.0

        if relative_drop > self.perf_tolerance:
            return SafetyReport(
                verdict=Verdict.FAIL_EXCEL,
                details={
                    "baseline_mean": baseline_mean,
                    "proposed_mean": proposed_mean,
                    "relative_drop": relative_drop,
                    "tolerance": self.perf_tolerance,
                },
                suggestions=[
                    f"Performance dropped by {relative_drop:.1%} "
                    f"(tolerance: {self.perf_tolerance:.1%}). "
                    "Revert or refine the mutation."
                ],
            )

        # --- Law III: Evolve (improvement) ----------------------------------
        if baseline_mean > 0:
            relative_gain = (proposed_mean - baseline_mean) / baseline_mean
        else:
            relative_gain = 1.0 if proposed_mean > 0 else 0.0

        if relative_gain < self.min_improvement:
            return SafetyReport(
                verdict=Verdict.FAIL_EVOLVE,
                details={
                    "baseline_mean": baseline_mean,
                    "proposed_mean": proposed_mean,
                    "relative_gain": relative_gain,
                    "min_improvement": self.min_improvement,
                },
                suggestions=[
                    f"Improvement of {relative_gain:.1%} is below the minimum "
                    f"{self.min_improvement:.1%}. Try a different mutation strategy."
                ],
            )

        # --- All laws passed ------------------------------------------------
        return SafetyReport(
            verdict=Verdict.PASS,
            details={
                "baseline_mean": baseline_mean,
                "proposed_mean": proposed_mean,
                "relative_gain": relative_gain,
                "error_rate": error_rate,
            },
        )

    # -- internal helpers ----------------------------------------------------
    @staticmethod
    def _evaluate(
        agent_fn: Callable[[Any], Any],
        inputs: List[Any],
        reward_fn: Callable[[Any, Any], float],
    ) -> List[float]:
        rewards = []
        for inp in inputs:
            try:
                out = agent_fn(inp)
                rewards.append(reward_fn(inp, out))
            except Exception:
                rewards.append(0.0)
        return rewards
