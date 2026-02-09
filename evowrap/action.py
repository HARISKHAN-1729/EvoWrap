
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .utils import Experience, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    """A callable tool the agent can invoke, with metadata for validation."""
    name: str
    fn: Callable[..., Any]
    description: str = ""
    param_names: Sequence[str] = ()

    def __call__(self, *args, **kwargs) -> Any:
        return self.fn(*args, **kwargs)


class ToolRegistry:
    """
    Typed registry of available tools.  The wrapped agent can discover, call,
    and validate tools through this interface.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info("Tool registered: %s", tool.name)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def call(self, name: str, *args, **kwargs) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        return tool(*args, **kwargs)


# ---------------------------------------------------------------------------
# Action trace — captures what the agent did and why
# ---------------------------------------------------------------------------
@dataclass
class ActionTrace:
    """Single entry in the agent's action history."""
    step: int
    input_data: Any
    reasoning: str            # chain-of-thought snippet
    action: Any               # raw agent output or tool call
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    result: Any = None
    reward: float = 0.0
    elapsed_sec: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Action Module — public API
# ---------------------------------------------------------------------------
class ActionModule:
    """
    Wraps the agent's raw *act* function with:
      1. Pre-execution tool validation.
      2. Post-execution result capture and reward assignment.
      3. Chain-of-thought logging for reflective optimisation.

    The module does NOT decide which action to take — that remains the agent's
    job.  It only *grounds* the action in a safe, traceable execution context.
    """

    def __init__(self) -> None:
        self.tools = ToolRegistry()
        self._history: List[ActionTrace] = []
        self._step = 0

    # -- execute an agent step -----------------------------------------------
    def execute(
        self,
        agent_fn: Callable[[Any], Any],
        input_data: Any,
        *,
        reasoning: str = "",
        reward_fn: Optional[Callable[[Any, Any], float]] = None,
    ) -> ActionTrace:
        """
        Run the agent, capture trace, compute reward.

        Parameters
        ----------
        agent_fn : callable
            The wrapped agent's `act(input) → output` function.
        input_data : any
            Current observation / prompt.
        reasoning : str
            Optional chain-of-thought string (from agent or prompt).
        reward_fn : callable, optional
            (input, output) → float.  If None, reward defaults to 0.
        """
        self._step += 1
        t0 = time.perf_counter()
        error = None
        result = None
        try:
            result = agent_fn(input_data)
        except Exception as exc:
            error = traceback.format_exc()
            logger.error("Agent execution error at step %d: %s", self._step, exc)

        elapsed = time.perf_counter() - t0

        reward = 0.0
        if reward_fn is not None and error is None:
            try:
                reward = reward_fn(input_data, result)
            except Exception as exc:
                logger.warning("Reward function failed: %s", exc)

        trace = ActionTrace(
            step=self._step,
            input_data=input_data,
            reasoning=reasoning,
            action=result,
            result=result,
            reward=reward,
            elapsed_sec=elapsed,
            error=error,
        )
        self._history.append(trace)
        return trace

    # -- tool execution (optional, for tool-using agents) --------------------
    def execute_tool(self, name: str, *args, **kwargs) -> Any:
        """Call a registered tool by name, with tracing."""
        t0 = time.perf_counter()
        try:
            result = self.tools.call(name, *args, **kwargs)
            logger.debug("Tool '%s' returned in %.3fs", name, time.perf_counter() - t0)
            return result
        except Exception as exc:
            logger.error("Tool '%s' failed: %s", name, exc)
            raise

    # -- trace utilities -----------------------------------------------------
    def recent_traces(self, n: int = 10) -> List[ActionTrace]:
        return self._history[-n:]

    def success_rate(self, last_n: int = 50) -> float:
        window = self._history[-last_n:]
        if not window:
            return 0.0
        return sum(1 for t in window if t.error is None) / len(window)

    def mean_reward(self, last_n: int = 50) -> float:
        window = self._history[-last_n:]
        if not window:
            return 0.0
        return sum(t.reward for t in window) / len(window)

    def to_experiences(self, last_n: int = 50) -> List[Experience]:
        """Convert recent traces to Experience objects for the memory module."""
        exps = []
        for t in self._history[-last_n:]:
            exps.append(
                Experience(
                    state=t.input_data,
                    action=t.action,
                    reward=t.reward,
                    metadata={"reasoning": t.reasoning, "error": t.error},
                )
            )
        return exps
