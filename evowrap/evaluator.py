"""
EvoWrap Evaluator — integrated metrics, benchmarks, and test suite.

Addresses survey critiques:
  - Lifelong survey Section 13: existing tools lack built-in benchmarks for
    long-term adaptation, relying on ad-hoc tests.
  - Self-evolving survey: incomplete coverage of safety metrics.
  - EvoWrap ships with a first-class evaluation suite that measures five
    dimensions after every evolution cycle: adaptation, forgetting, plasticity,
    safety compliance, and efficiency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .safety import SafetyReport, Verdict
from .utils import Timer, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric snapshots
# ---------------------------------------------------------------------------
@dataclass
class MetricSnapshot:
    """One evaluation at a point in time."""
    timestamp: float = 0.0
    adaptation_success_rate: float = 0.0   # % tasks solved after env change
    forgetting_ratio: float = 0.0          # (old_perf_post / old_perf_pre) - 1
    plasticity_steps: int = 0              # steps to reach threshold on new task
    safety_compliance: float = 0.0         # % evolutions passing Three Laws
    wall_time_sec: float = 0.0             # total compute time for the cycle
    memory_entries: int = 0                # long-term memory size
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark scenarios (synthetic continual-learning tasks)
# ---------------------------------------------------------------------------
class BanditTask:
    """
    Non-stationary multi-armed bandit.  Arms drift over time to simulate
    environment change — the simplest continual-learning benchmark.
    """

    def __init__(self, n_arms: int = 5, seed: int = 42):
        self.n_arms = n_arms
        self._rng = np.random.RandomState(seed)
        self._means = self._rng.randn(n_arms).astype(np.float32)
        self._step = 0
        self._phase = 0

    def reset(self, phase: int = 0) -> None:
        self._step = 0
        self._phase = phase
        self._means = self._rng.randn(self.n_arms).astype(np.float32) + phase

    def observe(self) -> np.ndarray:
        """Return noisy observation of arm means."""
        return self._means + self._rng.randn(self.n_arms).astype(np.float32) * 0.3

    def pull(self, arm: int) -> float:
        """Pull an arm, get reward."""
        arm = int(arm) % self.n_arms
        reward = float(self._means[arm] + self._rng.randn() * 0.5)
        self._step += 1
        return reward

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self._means))


class SequentialTaskBench:
    """
    Sequential task benchmark: agent must learn Task A, then Task B, then
    be re-tested on Task A.  Measures both adaptation and forgetting.
    """

    def __init__(self, n_arms: int = 5, n_phases: int = 3, steps_per_phase: int = 50):
        self.bandit = BanditTask(n_arms)
        self.n_phases = n_phases
        self.steps_per_phase = steps_per_phase

    def generate_inputs(self) -> List[List[np.ndarray]]:
        """Return a list of phases, each containing a list of observations."""
        phases = []
        for p in range(self.n_phases):
            self.bandit.reset(phase=p)
            obs = [self.bandit.observe() for _ in range(self.steps_per_phase)]
            phases.append(obs)
        return phases

    def evaluate_agent(
        self,
        agent_fn: Callable[[Any], Any],
        reward_fn: Callable[[Any, Any], float],
    ) -> Dict[str, float]:
        """
        Run the agent through all phases and compute metrics.
        """
        phases = self.generate_inputs()
        phase_rewards: List[List[float]] = []

        for p_idx, phase_obs in enumerate(phases):
            self.bandit.reset(phase=p_idx)
            rewards = []
            for obs in phase_obs:
                action = agent_fn(obs)
                arm = int(np.argmax(action)) if hasattr(action, '__len__') else int(action)
                r = self.bandit.pull(arm)
                rewards.append(r)
            phase_rewards.append(rewards)

        # Adaptation: mean reward on last phase
        adapt_mean = float(np.mean(phase_rewards[-1])) if phase_rewards else 0.0

        # Forgetting: compare first-phase reward to what agent gets when
        # re-tested on phase-0 distribution after learning later phases
        self.bandit.reset(phase=0)
        retest_rewards = []
        for _ in range(self.steps_per_phase):
            obs = self.bandit.observe()
            action = agent_fn(obs)
            arm = int(np.argmax(action)) if hasattr(action, '__len__') else int(action)
            r = self.bandit.pull(arm)
            retest_rewards.append(r)

        original_mean = float(np.mean(phase_rewards[0])) if phase_rewards[0] else 1e-9
        retest_mean = float(np.mean(retest_rewards))
        forgetting = (retest_mean / (original_mean + 1e-9)) - 1.0

        return {
            "adaptation_mean_reward": adapt_mean,
            "forgetting_ratio": forgetting,
            "phase_means": [float(np.mean(pr)) for pr in phase_rewards],
        }


# ---------------------------------------------------------------------------
# Evaluator — public API
# ---------------------------------------------------------------------------
class Evaluator:
    """
    Integrated evaluation suite.  Call `evaluate()` after each evolution
    cycle to get a full MetricSnapshot.  Call `run_benchmark()` for the
    built-in sequential-task test.

    Metrics (from lifelong survey Section 13):
        1. Adaptation Success Rate — % of test inputs where reward > threshold
        2. Forgetting Ratio — relative change in old-task performance
        3. Plasticity Score — steps to reach 80% of peak on a new task
        4. Safety Compliance — % of safety reports that passed
        5. Efficiency — wall time + memory footprint
    """

    def __init__(self, reward_threshold: float = 0.0):
        self.reward_threshold = reward_threshold
        self._history: List[MetricSnapshot] = []

    def evaluate(
        self,
        *,
        agent_fn: Callable[[Any], Any],
        test_inputs: List[Any],
        reward_fn: Callable[[Any, Any], float],
        old_task_inputs: Optional[List[Any]] = None,
        old_task_baseline_reward: float = 0.0,
        safety_reports: Optional[List[SafetyReport]] = None,
        memory_size: int = 0,
    ) -> MetricSnapshot:
        """
        Compute a full metric snapshot for the current agent state.
        """
        t0 = time.perf_counter()

        # 1. Adaptation success rate
        successes = 0
        rewards = []
        for inp in test_inputs:
            try:
                out = agent_fn(inp)
                r = reward_fn(inp, out)
                rewards.append(r)
                if r > self.reward_threshold:
                    successes += 1
            except Exception:
                rewards.append(0.0)
        adapt_rate = successes / max(len(test_inputs), 1)

        # 2. Forgetting ratio
        forgetting = 0.0
        if old_task_inputs and old_task_baseline_reward != 0:
            old_rewards = []
            for inp in old_task_inputs:
                try:
                    out = agent_fn(inp)
                    old_rewards.append(reward_fn(inp, out))
                except Exception:
                    old_rewards.append(0.0)
            old_mean = float(np.mean(old_rewards)) if old_rewards else 0.0
            forgetting = (old_mean / (old_task_baseline_reward + 1e-9)) - 1.0

        # 3. Plasticity (estimated from reward curve slope)
        if len(rewards) >= 2:
            # Steps until first time reward exceeds 80% of max
            peak = max(rewards) if rewards else 0
            target = 0.8 * peak
            plasticity = len(rewards)  # worst case
            for i, r in enumerate(rewards):
                if r >= target:
                    plasticity = i + 1
                    break
        else:
            plasticity = 0

        # 4. Safety compliance
        if safety_reports:
            passed = sum(1 for r in safety_reports if r.passed)
            safety_rate = passed / len(safety_reports)
        else:
            safety_rate = 1.0

        elapsed = time.perf_counter() - t0

        snap = MetricSnapshot(
            timestamp=time.time(),
            adaptation_success_rate=adapt_rate,
            forgetting_ratio=forgetting,
            plasticity_steps=plasticity,
            safety_compliance=safety_rate,
            wall_time_sec=elapsed,
            memory_entries=memory_size,
            extra={"mean_reward": float(np.mean(rewards)) if rewards else 0.0},
        )
        self._history.append(snap)
        return snap

    def run_benchmark(
        self,
        agent_fn: Callable[[Any], Any],
        reward_fn: Callable[[Any, Any], float],
        n_arms: int = 5,
        n_phases: int = 3,
        steps_per_phase: int = 50,
    ) -> Dict[str, Any]:
        """Run the built-in sequential-task benchmark."""
        bench = SequentialTaskBench(n_arms, n_phases, steps_per_phase)
        return bench.evaluate_agent(agent_fn, reward_fn)

    def report(self, last_n: int = 5) -> str:
        """Human-readable summary of recent evaluations."""
        lines = ["=" * 60, " EvoWrap Evaluation Report", "=" * 60]
        for snap in self._history[-last_n:]:
            lines.append(
                f"  Adaptation:  {snap.adaptation_success_rate:.1%}  |  "
                f"Forgetting: {snap.forgetting_ratio:+.2%}  |  "
                f"Plasticity: {snap.plasticity_steps} steps  |  "
                f"Safety: {snap.safety_compliance:.0%}  |  "
                f"Time: {snap.wall_time_sec:.3f}s  |  "
                f"Memory: {snap.memory_entries}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    @property
    def history(self) -> List[MetricSnapshot]:
        return list(self._history)
