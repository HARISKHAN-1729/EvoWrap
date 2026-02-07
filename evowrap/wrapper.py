"""
EvoWrap Core Wrapper — the @evowrap decorator and EvoWrapAgent class.

This is the user-facing integration layer.  A developer wraps their existing
agent in one line:

    @evowrap(genome={"temperature": 0.7}, reward_fn=my_scorer)
    def my_agent(input_data):
        return call_llm(input_data)

The wrapped agent now:
  - Perceives multi-modal inputs and detects distribution drift.
  - Stores every interaction in hybrid memory with replay.
  - Tracks action traces with chain-of-thought.
  - Auto-evolves via evolutionary + RL optimisation when performance drops.
  - Enforces Three Laws safety on every evolution.
  - Logs evaluation metrics after each cycle.

This addresses the core critique: no existing framework can wrap an *arbitrary*
agent (LLM, RL, rule-based, symbolic) with continual learning + safety in a
single decorator.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
import os
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .action import ActionModule
from .evaluator import Evaluator
from .memory import MemoryModule
from .optimizer import Genome, OptimizerEngine
from .perception import PerceptionModule
from .safety import SafetyChecker
from .utils import Experience, get_logger, text_embed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# EvoWrapAgent — the fully-assembled runtime
# ---------------------------------------------------------------------------
class EvoWrapAgent:
    """
    Wraps any `agent_fn(input) → output` with the full EvoWrap pipeline.

    Parameters
    ----------
    agent_fn : callable
        The original agent to wrap.
    genome : dict
        Initial mutable parameters (evolve-able).
    build_fn : callable, optional
        (Genome) → agent_fn.  If None, mutations change `genome.params`
        but the same `agent_fn` is used (suitable for stateless agents
        where params are read from the genome at call time).
    reward_fn : callable, optional
        (input, output) → float.  Defaults to always 0.
    perception : PerceptionModule, optional
    memory : MemoryModule, optional
    action : ActionModule, optional
    safety : SafetyChecker, optional
    optimizer : OptimizerEngine, optional
    evaluator : Evaluator, optional
    evolve_every : int
        Run an evolution check every N steps.
    auto_evolve : bool
        If True, automatically evolve when drift or low reward is detected.
    rollout_fraction : float
        Fraction of calls using new genome after evolution (A/B testing).
        1.0 = instant swap (default). < 1.0 = gradual rollout.
    """

    def __init__(
        self,
        agent_fn: Callable[[Any], Any],
        *,
        genome: Optional[Dict[str, Any]] = None,
        build_fn: Optional[Callable[[Genome], Callable[[Any], Any]]] = None,
        reward_fn: Optional[Callable[[Any, Any], float]] = None,
        perception: Optional[PerceptionModule] = None,
        memory: Optional[MemoryModule] = None,
        action: Optional[ActionModule] = None,
        safety: Optional[SafetyChecker] = None,
        optimizer: Optional[OptimizerEngine] = None,
        evaluator: Optional[Evaluator] = None,
        evolve_every: int = 50,
        auto_evolve: bool = True,
        rollout_fraction: float = 1.0,
    ):
        self._original_fn = agent_fn
        self._current_fn = agent_fn
        self._previous_fn: Optional[Callable] = None  # for A/B rollout
        self._genome = Genome(params=genome or {})
        self._build_fn = build_fn or (lambda g: agent_fn)
        self._reward_fn = reward_fn or (lambda inp, out: 0.0)

        # Sub-modules (create defaults if not provided)
        self.perception = perception or PerceptionModule()
        self.memory = memory or MemoryModule()
        self.action = action or ActionModule()
        self.safety = safety or SafetyChecker()
        self.optimizer = optimizer or OptimizerEngine(self.safety)
        self.evaluator = evaluator or Evaluator()

        self._evolve_every = evolve_every
        self._auto_evolve = auto_evolve
        self._step = 0
        self._task_id = "default"
        self._test_input_buffer: List[Any] = []
        self._evolution_count = 0
        self._all_safety_reports = []

        # A/B rollout
        self._rollout_fraction = rollout_fraction

        # Forgetting measurement — per-task tracking
        self._task_input_history: Dict[str, List[Any]] = defaultdict(list)
        self._task_reward_history: Dict[str, float] = {}

    # -- main call -----------------------------------------------------------
    def __call__(self, input_data: Any) -> Any:
        """
        The wrapped agent call.  This is the hot path:
          1. Perceive → 2. Recall → 3. Act → 4. Store → 5. Maybe evolve
        """
        self._step += 1

        # 1. Perceive: convert input to embedding, check drift
        if isinstance(input_data, dict):
            perception_input = input_data
        elif isinstance(input_data, np.ndarray) and input_data.ndim >= 2:
            perception_input = {"image": input_data}
        elif isinstance(input_data, str):
            perception_input = {"text": input_data}
        else:
            perception_input = {"text": str(input_data)}

        embedding, drifted = self.perception.perceive(perception_input)

        # 2. Recall relevant experiences from long-term memory
        relevant = self.memory.recall(embedding, top_k=3)

        # 3. Act: pick agent fn (A/B rollout support)
        active_fn = self._pick_agent_fn()

        trace = self.action.execute(
            active_fn,
            input_data,
            reasoning=f"step={self._step}, drift={drifted}, "
                      f"recalled={len(relevant)} experiences",
            reward_fn=self._reward_fn,
        )

        # 4. Store the experience
        exp = Experience(
            state=input_data,
            action=trace.action,
            reward=trace.reward,
            embedding=embedding,
            metadata={"task_id": self._task_id, "step": self._step},
        )
        self.memory.store(exp, embedding=embedding, task_id=self._task_id)

        # Track per-task inputs for forgetting measurement
        task_inputs = self._task_input_history[self._task_id]
        task_inputs.append(input_data)
        if len(task_inputs) > 50:
            self._task_input_history[self._task_id] = task_inputs[-50:]

        # Keep a rolling window of inputs for evolution testing
        self._test_input_buffer.append(input_data)
        if len(self._test_input_buffer) > 200:
            self._test_input_buffer = self._test_input_buffer[-100:]

        # 5. Maybe evolve
        if (
            self._auto_evolve
            and self._step % self._evolve_every == 0
            and len(self._test_input_buffer) >= 10
        ):
            recent_reward = self.action.mean_reward(last_n=self._evolve_every)
            if self.optimizer.should_evolve(recent_reward, drifted):
                self.evolve()

        return trace.action

    # -- A/B rollout agent selection -----------------------------------------
    def _pick_agent_fn(self) -> Callable:
        """Pick agent fn based on rollout fraction (A/B testing)."""
        if (
            self._rollout_fraction < 1.0
            and self._previous_fn is not None
            and random.random() >= self._rollout_fraction
        ):
            return self._previous_fn
        return self._current_fn

    def promote(self) -> None:
        """Fully switch to the new agent (end A/B rollout)."""
        self._previous_fn = None
        self._rollout_fraction = 1.0
        logger.info("A/B rollout promoted — new agent is now sole agent")

    def rollback(self) -> None:
        """Revert to the old agent (reject A/B rollout)."""
        if self._previous_fn is not None:
            self._current_fn = self._previous_fn
            self._previous_fn = None
            self._rollout_fraction = 1.0
            logger.info("A/B rollout rolled back — reverted to previous agent")

    # -- manual evolution trigger --------------------------------------------
    def evolve(self) -> Dict[str, Any]:
        """
        Run a full evolution cycle and return a summary dict.
        """
        self._evolution_count += 1
        logger.info("=== Evolution cycle %d ===", self._evolution_count)

        test_inputs = self._test_input_buffer[-50:]
        forgetting_penalty = self.memory.forgetting_penalty(self._task_id)

        new_genome, reports = self.optimizer.run_cycle(
            baseline_genome=self._genome,
            build_fn=self._build_fn,
            reward_fn=self._reward_fn,
            test_inputs=test_inputs,
            forgetting_penalty=forgetting_penalty,
        )
        self._all_safety_reports.extend(reports)

        accepted = any(r.passed for r in reports)
        if accepted:
            self._genome = new_genome
            # A/B rollout: keep old fn if rollout_fraction < 1.0
            if self._rollout_fraction < 1.0:
                self._previous_fn = self._current_fn
            self._current_fn = self._build_fn(new_genome)
            logger.info("Evolution accepted — agent updated (fitness=%.4f)", new_genome.fitness)
        else:
            logger.info("Evolution rejected — agent unchanged")

        # Gather old task inputs for forgetting measurement
        old_inputs: List[Any] = []
        old_baseline = 0.0
        for tid, inputs in self._task_input_history.items():
            if tid != self._task_id:
                old_inputs.extend(inputs[-20:])
        if old_inputs:
            old_baselines = [v for k, v in self._task_reward_history.items() if k != self._task_id]
            old_baseline = float(np.mean(old_baselines)) if old_baselines else 0.0

        # Evaluate post-evolution
        snap = self.evaluator.evaluate(
            agent_fn=self._current_fn,
            test_inputs=test_inputs,
            reward_fn=self._reward_fn,
            safety_reports=reports,
            memory_size=len(self.memory.long_term),
            old_task_inputs=old_inputs if old_inputs else None,
            old_task_baseline_reward=old_baseline,
        )

        return {
            "cycle": self._evolution_count,
            "accepted": accepted,
            "fitness": new_genome.fitness,
            "metrics": {
                "adaptation": snap.adaptation_success_rate,
                "forgetting": snap.forgetting_ratio,
                "plasticity": snap.plasticity_steps,
                "safety": snap.safety_compliance,
            },
        }

    # -- task management (for continual learning) ----------------------------
    def set_task(self, task_id: str) -> None:
        """Switch to a new task context (keeps old memories for replay)."""
        # Store mean reward for current task before switching
        if self._task_id != task_id:
            recent_reward = self.action.mean_reward(last_n=self._evolve_every)
            self._task_reward_history[self._task_id] = recent_reward
            logger.info(
                "Task switch: %s → %s (stored reward=%.3f for %s)",
                self._task_id, task_id, recent_reward, self._task_id,
            )
        self._task_id = task_id

    # -- checkpoint/restore --------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """
        Save full agent state to a directory.

        Creates:
          path/genome.json     — genome params, fitness, generation, lineage
          path/memory/         — memory module state
          path/state.json      — step count, task history, evolution count, etc.
        """
        os.makedirs(path, exist_ok=True)

        # Genome
        genome_data = {
            "params": self._genome.params,
            "fitness": self._genome.fitness,
            "generation": self._genome.generation,
            "lineage": self._genome.lineage,
        }
        with open(os.path.join(path, "genome.json"), "w") as f:
            json.dump(genome_data, f, indent=2, default=str)

        # Memory
        self.memory.save(os.path.join(path, "memory"))

        # Agent state
        state = {
            "step": self._step,
            "task_id": self._task_id,
            "evolution_count": self._evolution_count,
            "evolve_every": self._evolve_every,
            "auto_evolve": self._auto_evolve,
            "rollout_fraction": self._rollout_fraction,
            "task_reward_history": self._task_reward_history,
            "evolution_history": self.optimizer.evolution_history,
        }
        with open(os.path.join(path, "state.json"), "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info("Checkpoint saved to %s", path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        agent_fn: Callable[[Any], Any],
        build_fn: Callable[[Genome], Callable[[Any], Any]],
        reward_fn: Callable[[Any, Any], float],
    ) -> "EvoWrapAgent":
        """
        Restore an EvoWrapAgent from a saved checkpoint.

        Parameters
        ----------
        path : str
            Directory containing the checkpoint files.
        agent_fn, build_fn, reward_fn : callables
            Must be provided since functions can't be serialized.
        """
        # Load genome
        with open(os.path.join(path, "genome.json"), "r") as f:
            genome_data = json.load(f)
        genome = Genome(
            params=genome_data["params"],
            fitness=genome_data["fitness"],
            generation=genome_data["generation"],
            lineage=genome_data["lineage"],
        )

        # Load state
        with open(os.path.join(path, "state.json"), "r") as f:
            state = json.load(f)

        # Load memory
        memory = MemoryModule.load(os.path.join(path, "memory"))

        # Reconstruct agent
        agent = cls(
            agent_fn=build_fn(genome),
            genome=genome.params,
            build_fn=build_fn,
            reward_fn=reward_fn,
            memory=memory,
            evolve_every=state["evolve_every"],
            auto_evolve=state["auto_evolve"],
            rollout_fraction=state.get("rollout_fraction", 1.0),
        )
        agent._genome = genome
        agent._step = state["step"]
        agent._task_id = state["task_id"]
        agent._evolution_count = state["evolution_count"]
        agent._task_reward_history = state.get("task_reward_history", {})

        logger.info("Checkpoint loaded from %s (step=%d)", path, agent._step)
        return agent

    # -- accessors -----------------------------------------------------------
    @property
    def genome(self) -> Genome:
        return self._genome

    @property
    def step(self) -> int:
        return self._step

    def report(self) -> str:
        return self.evaluator.report()

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "evolution_cycles": self._evolution_count,
            "memory": self.memory.stats,
            "success_rate": self.action.success_rate(),
            "mean_reward": self.action.mean_reward(),
            "drift_score": self.perception.drift_score,
        }


# ---------------------------------------------------------------------------
# AsyncEvoWrapAgent — async variant for I/O-bound agent functions
# ---------------------------------------------------------------------------
class AsyncEvoWrapAgent(EvoWrapAgent):
    """
    Async variant of EvoWrapAgent for agents that use async I/O
    (e.g., async HTTP calls to LLM APIs).

    Usage:
        agent = AsyncEvoWrapAgent(my_async_fn, genome={...}, ...)
        result = await agent(input_data)
    """

    async def __call__(self, input_data: Any) -> Any:  # type: ignore[override]
        """
        Async wrapped agent call. Same pipeline as sync but awaits agent_fn.
        """
        self._step += 1

        # 1. Perceive
        if isinstance(input_data, dict):
            perception_input = input_data
        elif isinstance(input_data, np.ndarray) and input_data.ndim >= 2:
            perception_input = {"image": input_data}
        elif isinstance(input_data, str):
            perception_input = {"text": input_data}
        else:
            perception_input = {"text": str(input_data)}

        embedding, drifted = self.perception.perceive(perception_input)

        # 2. Recall
        relevant = self.memory.recall(embedding, top_k=3)

        # 3. Act — await the async agent function
        active_fn = self._pick_agent_fn()
        import time as _time
        t0 = _time.perf_counter()
        error = None
        result = None
        try:
            result = await active_fn(input_data)
        except Exception as exc:
            import traceback
            error = traceback.format_exc()
            logger.error("Async agent error at step %d: %s", self._step, exc)

        elapsed = _time.perf_counter() - t0

        reward = 0.0
        if error is None and self._reward_fn is not None:
            try:
                reward = self._reward_fn(input_data, result)
            except Exception:
                pass

        # Record in action module for stats
        from .action import ActionTrace
        trace = ActionTrace(
            step=self._step,
            input_data=input_data,
            reasoning=f"async step={self._step}, drift={drifted}",
            action=result,
            result=result,
            reward=reward,
            elapsed_sec=elapsed,
            error=error,
        )
        self.action._history.append(trace)
        self.action._step += 1

        # 4. Store
        exp = Experience(
            state=input_data,
            action=result,
            reward=reward,
            embedding=embedding,
            metadata={"task_id": self._task_id, "step": self._step},
        )
        self.memory.store(exp, embedding=embedding, task_id=self._task_id)

        # Track per-task inputs
        task_inputs = self._task_input_history[self._task_id]
        task_inputs.append(input_data)
        if len(task_inputs) > 50:
            self._task_input_history[self._task_id] = task_inputs[-50:]

        self._test_input_buffer.append(input_data)
        if len(self._test_input_buffer) > 200:
            self._test_input_buffer = self._test_input_buffer[-100:]

        # 5. Maybe evolve (synchronous — evolution is compute-bound)
        if (
            self._auto_evolve
            and self._step % self._evolve_every == 0
            and len(self._test_input_buffer) >= 10
        ):
            recent_reward = self.action.mean_reward(last_n=self._evolve_every)
            if self.optimizer.should_evolve(recent_reward, drifted):
                self.evolve()

        return result


# ---------------------------------------------------------------------------
# @evowrap decorator — the simplest integration path
# ---------------------------------------------------------------------------
def evowrap(
    _fn: Optional[Callable] = None,
    *,
    genome: Optional[Dict[str, Any]] = None,
    build_fn: Optional[Callable[[Genome], Callable[[Any], Any]]] = None,
    reward_fn: Optional[Callable[[Any, Any], float]] = None,
    evolve_every: int = 50,
    auto_evolve: bool = True,
    **kwargs,
):
    """
    Decorator to wrap any agent function with EvoWrap.

    Usage:
        @evowrap(genome={"temp": 0.7}, reward_fn=scorer)
        def my_agent(inp):
            return do_stuff(inp)

        # my_agent is now an EvoWrapAgent instance
        result = my_agent("hello")
    """
    def decorator(fn: Callable) -> EvoWrapAgent:
        return EvoWrapAgent(
            fn,
            genome=genome,
            build_fn=build_fn,
            reward_fn=reward_fn,
            evolve_every=evolve_every,
            auto_evolve=auto_evolve,
            **kwargs,
        )

    if _fn is not None:
        # Called without arguments: @evowrap
        return decorator(_fn)
    # Called with arguments: @evowrap(...)
    return decorator


def async_evowrap(
    _fn: Optional[Callable] = None,
    *,
    genome: Optional[Dict[str, Any]] = None,
    build_fn: Optional[Callable[[Genome], Callable[[Any], Any]]] = None,
    reward_fn: Optional[Callable[[Any, Any], float]] = None,
    evolve_every: int = 50,
    auto_evolve: bool = True,
    **kwargs,
):
    """
    Decorator to wrap an async agent function with EvoWrap.

    Usage:
        @async_evowrap(genome={"temp": 0.7}, reward_fn=scorer)
        async def my_agent(inp):
            return await llm_call(inp)

        result = await my_agent("hello")
    """
    def decorator(fn: Callable) -> AsyncEvoWrapAgent:
        return AsyncEvoWrapAgent(
            fn,
            genome=genome,
            build_fn=build_fn,
            reward_fn=reward_fn,
            evolve_every=evolve_every,
            auto_evolve=auto_evolve,
            **kwargs,
        )

    if _fn is not None:
        return decorator(_fn)
    return decorator
