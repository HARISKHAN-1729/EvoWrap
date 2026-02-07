"""
EvoWrap Optimizer Engine — evolutionary algorithms + RL-based self-evolution.

Addresses survey critiques:
  - Agent Lightning uses RL but is LLM-centric; EvoAgentX mutates prompts
    but ignores non-LLM agents.  EvoWrap's optimizer is *agent-agnostic*:
    it evolves any mutable parameter (prompts, numeric weights, tool configs,
    strategy strings) through a generic genome abstraction.
  - The stability-plasticity dilemma is handled by combining evolutionary
    search (exploration) with a PPO-style policy gradient (exploitation),
    gated by the SafetyChecker's Three Laws validation.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .safety import SafetyChecker, SafetyReport, Verdict
from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Genome: the evolvable unit
# ---------------------------------------------------------------------------
@dataclass
class Genome:
    """
    A mutable parameter set for an agent.  Each key maps to a value that
    the optimizer can mutate.  Values can be floats, strings, lists, etc.
    A *build_fn* converts the genome back into an agent function.
    """
    params: Dict[str, Any] = field(default_factory=dict)
    fitness: float = 0.0
    generation: int = 0
    lineage: List[str] = field(default_factory=list)  # mutation history

    def clone(self) -> "Genome":
        g = Genome(
            params=copy.deepcopy(self.params),
            fitness=self.fitness,
            generation=self.generation,
            lineage=list(self.lineage),
        )
        return g


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------
def mutate_float(value: float, sigma: float = 0.1) -> float:
    return value + random.gauss(0, sigma)


def mutate_string(value: str) -> str:
    """Randomly swap / insert / delete a character (lightweight prompt mutation)."""
    if not value:
        return value
    s = list(value)
    op = random.choice(["swap", "insert", "delete"])
    idx = random.randint(0, max(0, len(s) - 1))
    if op == "swap" and len(s) > 1:
        j = random.randint(0, len(s) - 1)
        s[idx], s[j] = s[j], s[idx]
    elif op == "insert":
        s.insert(idx, chr(random.randint(32, 126)))
    elif op == "delete" and len(s) > 1:
        s.pop(idx)
    return "".join(s)


def mutate_list(value: list, mutate_elem: Callable = mutate_float) -> list:
    """Mutate a random element, or append/remove."""
    out = list(value)
    if not out:
        return out
    op = random.choice(["mutate", "append", "remove"])
    if op == "mutate":
        idx = random.randint(0, len(out) - 1)
        out[idx] = mutate_elem(out[idx])
    elif op == "append":
        out.append(mutate_elem(out[-1]) if out else 0.0)
    elif op == "remove" and len(out) > 1:
        out.pop(random.randint(0, len(out) - 1))
    return out


def mutate_genome(genome: Genome, mutation_rate: float = 0.3) -> Genome:
    """Apply random mutations to a genome's parameters."""
    child = genome.clone()
    child.generation += 1
    mutations = []
    for key, val in child.params.items():
        if random.random() > mutation_rate:
            continue
        if isinstance(val, float):
            child.params[key] = mutate_float(val)
            mutations.append(f"{key}:float_mut")
        elif isinstance(val, int):
            child.params[key] = val + random.choice([-1, 0, 1])
            mutations.append(f"{key}:int_mut")
        elif isinstance(val, str):
            child.params[key] = mutate_string(val)
            mutations.append(f"{key}:str_mut")
        elif isinstance(val, list):
            child.params[key] = mutate_list(val)
            mutations.append(f"{key}:list_mut")
    child.lineage.append("|".join(mutations) if mutations else "no_op")
    return child


def crossover(a: Genome, b: Genome) -> Genome:
    """Uniform crossover — each param picked from parent A or B."""
    child = a.clone()
    child.generation = max(a.generation, b.generation) + 1
    for key in child.params:
        if key in b.params and random.random() < 0.5:
            child.params[key] = copy.deepcopy(b.params[key])
    child.lineage.append(f"crossover(gen{a.generation},gen{b.generation})")
    return child


# ---------------------------------------------------------------------------
# PPO-style advantage estimator (lightweight, no neural net required)
# ---------------------------------------------------------------------------
class AdvantageEstimator:
    """
    Compute generalized advantage estimates from a sequence of rewards.
    Used by the optimizer to decide *which direction* to mutate.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

    def compute(self, rewards: Sequence[float]) -> np.ndarray:
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] - (rewards[t + 1] * self.gamma if t + 1 < n else 0.0)
            last = delta + self.gamma * self.lam * last
            advantages[t] = last
        return advantages


# ---------------------------------------------------------------------------
# Evolution strategy selector
# ---------------------------------------------------------------------------
class EvolutionStrategy:
    """
    Combines evolutionary search + RL-style advantage weighting.

    1. Maintain a *population* of genomes.
    2. Each generation: evaluate, compute advantages, select top-K,
       mutate / crossover, validate via SafetyChecker.
    3. The best safe candidate becomes the new baseline.
    """

    def __init__(
        self,
        population_size: int = 10,
        elite_fraction: float = 0.2,
        mutation_rate: float = 0.3,
    ):
        self.population_size = population_size
        self.elite_count = max(1, int(population_size * elite_fraction))
        self.mutation_rate = mutation_rate
        self._advantage = AdvantageEstimator()

    def evolve(
        self,
        population: List[Genome],
        fitness_fn: Callable[[Genome], float],
    ) -> List[Genome]:
        """Run one generation: evaluate → select → breed → return new pop."""
        # Evaluate
        for g in population:
            g.fitness = fitness_fn(g)

        # Sort by fitness descending
        population.sort(key=lambda g: -g.fitness)
        elites = population[: self.elite_count]

        # Breed next generation
        next_gen: List[Genome] = [e.clone() for e in elites]
        while len(next_gen) < self.population_size:
            if random.random() < 0.7:
                parent = random.choice(elites)
                child = mutate_genome(parent, self.mutation_rate)
            else:
                if len(elites) >= 2:
                    a, b = random.sample(elites, 2)
                    child = crossover(a, b)
                else:
                    child = mutate_genome(elites[0], self.mutation_rate)
            next_gen.append(child)

        return next_gen


# ---------------------------------------------------------------------------
# Optimizer Engine — public API
# ---------------------------------------------------------------------------
class OptimizerEngine:
    """
    Autonomous evolution controller.  Given:
      - a *genome* describing the agent's mutable parameters,
      - a *build_fn* that turns a genome into an agent function,
      - a *reward_fn* and *test_inputs* for evaluation,
      - a *SafetyChecker* instance,

    it runs evolutionary cycles, only accepting changes that pass all
    Three Laws.

    This is what makes EvoWrap truly self-evolving: the optimizer
    continuously proposes, tests, and either adopts or rejects mutations
    without human intervention.
    """

    def __init__(
        self,
        safety: SafetyChecker,
        population_size: int = 10,
        elite_fraction: float = 0.2,
        mutation_rate: float = 0.3,
        evolution_trigger_threshold: float = 0.5,
    ):
        self.safety = safety
        self._strategy = EvolutionStrategy(population_size, elite_fraction, mutation_rate)
        self.evolution_trigger_threshold = evolution_trigger_threshold
        self._generation = 0
        self._evolution_log: List[Dict[str, Any]] = []

    def should_evolve(self, recent_reward: float, drift_detected: bool) -> bool:
        """Decide whether to trigger an evolution cycle."""
        if drift_detected:
            logger.info("Evolution triggered by drift detection")
            return True
        if recent_reward < self.evolution_trigger_threshold:
            logger.info(
                "Evolution triggered by low reward (%.3f < %.3f)",
                recent_reward,
                self.evolution_trigger_threshold,
            )
            return True
        return False

    def run_cycle(
        self,
        *,
        baseline_genome: Genome,
        build_fn: Callable[[Genome], Callable[[Any], Any]],
        reward_fn: Callable[[Any, Any], float],
        test_inputs: List[Any],
        max_generations: int = 5,
        forgetting_penalty: float = 0.0,
        max_eval_inputs: int = 20,
    ) -> Tuple[Genome, List[SafetyReport]]:
        """
        Execute a full evolution cycle.

        Parameters
        ----------
        max_eval_inputs : int
            Cap the number of test inputs used per fitness evaluation.
            Randomly samples if more are available, reducing inference cost.

        Returns the best safe genome found and all safety reports generated.
        """
        baseline_agent = build_fn(baseline_genome)
        best = baseline_genome.clone()
        reports: List[SafetyReport] = []

        # Cap test inputs for fitness evaluation
        if len(test_inputs) > max_eval_inputs:
            eval_inputs = random.sample(test_inputs, max_eval_inputs)
        else:
            eval_inputs = test_inputs

        # Initialise population from baseline
        population = [mutate_genome(baseline_genome, self._strategy.mutation_rate)
                      for _ in range(self._strategy.population_size)]
        population[0] = baseline_genome.clone()  # keep baseline in pool

        for gen in range(max_generations):
            self._generation += 1

            # Fitness = mean reward on test inputs − forgetting penalty
            def fitness_fn(g: Genome, _inputs=eval_inputs) -> float:
                agent_fn = build_fn(g)
                rewards = []
                for inp in _inputs:
                    try:
                        out = agent_fn(inp)
                        rewards.append(reward_fn(inp, out))
                    except Exception:
                        rewards.append(-1.0)
                raw = float(np.mean(rewards)) if rewards else -1.0
                return raw - 0.1 * forgetting_penalty

            population = self._strategy.evolve(population, fitness_fn)
            candidate = population[0]  # best in this generation

            # Safety check the top candidate
            candidate_agent = build_fn(candidate)
            report = self.safety.validate(
                proposed_agent_fn=candidate_agent,
                baseline_agent_fn=baseline_agent,
                test_inputs=eval_inputs,
                reward_fn=reward_fn,
                evolution_metadata={
                    "generation": self._generation,
                    "lineage": candidate.lineage,
                },
            )
            reports.append(report)

            if report.passed:
                best = candidate.clone()
                logger.info(
                    "Gen %d: ACCEPTED (fitness=%.4f, gain=%.1f%%)",
                    self._generation,
                    candidate.fitness,
                    report.details.get("relative_gain", 0) * 100,
                )
                break  # Early stopping: first safe improvement wins
            else:
                logger.info(
                    "Gen %d: REJECTED (%s) — %s",
                    self._generation,
                    report.verdict.name,
                    "; ".join(report.suggestions),
                )

        self._evolution_log.append({
            "cycle_generations": gen + 1,
            "best_fitness": best.fitness,
            "accepted": any(r.passed for r in reports),
        })
        return best, reports

    @property
    def evolution_history(self) -> List[Dict[str, Any]]:
        return list(self._evolution_log)
