"""
EvoWrap — Agent-Agnostic Self-Evolving Framework
=================================================

Transform any AI agent into a continually learning, self-evolving system
with a single decorator::

    from evowrap import evowrap

    @evowrap(genome={"temp": 0.7}, reward_fn=my_scorer)
    def my_agent(inp):
        return llm_call(inp)

Key modules:
    - perception  : Multi-modal input fusion & drift detection
    - memory      : Hybrid short/long-term store with replay & regularisation
    - action      : Tool grounding & chain-of-thought tracing
    - optimizer   : Evolutionary + RL-based self-evolution
    - safety      : Three Laws enforcement (Endure → Excel → Evolve)
    - evaluator   : Integrated metrics & benchmarks
    - wrapper     : @evowrap decorator & EvoWrapAgent class
"""

__version__ = "0.2.0"

# Public API — everything a user needs from a single import
from .wrapper import EvoWrapAgent, evowrap, AsyncEvoWrapAgent, async_evowrap
from .perception import PerceptionModule, Modality, ModalityAdapter, DriftDetector
from .memory import MemoryModule, VectorIndex, ImportanceTracker, FaissIndex, make_vector_index
from .action import ActionModule, Tool, ToolRegistry
from .optimizer import OptimizerEngine, Genome, EvolutionStrategy
from .safety import SafetyChecker, SafetyReport, Verdict, Guardrail
from .evaluator import Evaluator, MetricSnapshot, SequentialTaskBench
from .utils import Experience, RingBuffer, enable_json_logging, MetricsExporter, JsonFormatter

__all__ = [
    # Core
    "evowrap",
    "EvoWrapAgent",
    "async_evowrap",
    "AsyncEvoWrapAgent",
    # Perception
    "PerceptionModule",
    "Modality",
    "ModalityAdapter",
    "DriftDetector",
    # Memory
    "MemoryModule",
    "VectorIndex",
    "FaissIndex",
    "make_vector_index",
    "ImportanceTracker",
    # Action
    "ActionModule",
    "Tool",
    "ToolRegistry",
    # Optimizer
    "OptimizerEngine",
    "Genome",
    "EvolutionStrategy",
    # Safety
    "SafetyChecker",
    "SafetyReport",
    "Verdict",
    "Guardrail",
    # Evaluator
    "Evaluator",
    "MetricSnapshot",
    "SequentialTaskBench",
    # Utils
    "Experience",
    "RingBuffer",
    "enable_json_logging",
    "MetricsExporter",
    "JsonFormatter",
]
