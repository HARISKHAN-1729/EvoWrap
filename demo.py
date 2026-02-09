#!/usr/bin/env python3

import numpy as np
from evowrap import (
    EvoWrapAgent,
    Evaluator,
    Genome,
    MemoryModule,
    OptimizerEngine,
    PerceptionModule,
    SafetyChecker,
    SequentialTaskBench,
    evowrap,
)


# ===== 1. Define a dummy agent =============================================
# A simple epsilon-greedy bandit agent.  Its behaviour depends on two
# genome parameters: `epsilon` (exploration rate) and `bias` (arm preference).

N_ARMS = 5

def make_agent(genome: Genome):
    """Build an agent function from a genome (the optimizer calls this)."""
    epsilon = float(genome.params.get("epsilon", 0.3))
    bias = np.array(genome.params.get("bias", [0.0] * N_ARMS), dtype=np.float32)

    def agent_fn(observation):
        obs = np.asarray(observation, dtype=np.float32).ravel()[:N_ARMS]
        if np.random.random() < epsilon:
            return np.random.randint(N_ARMS)
        # Exploit: pick arm with highest (obs + bias)
        scores = obs + bias[:len(obs)]
        return int(np.argmax(scores))
    return agent_fn


# ===== 2. Define the reward function =======================================
# For the bandit: reward = value of the arm pulled.
# We use a global bandit instance that the demo mutates across phases.

class BanditEnv:
    def __init__(self, n_arms=N_ARMS, seed=42):
        self.n_arms = n_arms
        self._rng = np.random.RandomState(seed)
        self._means = self._rng.randn(n_arms).astype(np.float32)

    def observe(self):
        return self._means + self._rng.randn(self.n_arms).astype(np.float32) * 0.3

    def pull(self, arm):
        arm = int(arm) % self.n_arms
        return float(self._means[arm] + self._rng.randn() * 0.5)

    def shift(self, delta=2.0):
        """Simulate environment change (distribution drift)."""
        self._means = self._rng.randn(self.n_arms).astype(np.float32) + delta

    @property
    def optimal_arm(self):
        return int(np.argmax(self._means))


env = BanditEnv()

def reward_fn(observation, action):
    return env.pull(action)


# ===== 3. Create the EvoWrap agent =========================================
initial_genome = {
    "epsilon": 0.3,
    "bias": [0.0] * N_ARMS,
}

agent = EvoWrapAgent(
    agent_fn=make_agent(Genome(params=initial_genome)),
    genome=initial_genome,
    build_fn=make_agent,
    reward_fn=reward_fn,
    evolve_every=30,          # check evolution every 30 steps
    auto_evolve=True,
    safety=SafetyChecker(
        perf_tolerance=0.10,   # allow up to 10% drop
        min_improvement=0.005, # require at least 0.5% gain
        crash_sim_trials=10,
        max_error_rate=0.3,
    ),
    optimizer=OptimizerEngine(
        safety=SafetyChecker(perf_tolerance=0.10, min_improvement=0.005),
        population_size=8,
        mutation_rate=0.4,
        evolution_trigger_threshold=0.3,
    ),
)


# ===== 4. Run the demo =====================================================
def run_phase(name, steps, task_id):
    print(f"\n{'='*60}")
    print(f"  Phase: {name}  ({steps} steps, task={task_id})")
    print(f"  Optimal arm: {env.optimal_arm}  |  Means: {env._means.round(2)}")
    print(f"{'='*60}")

    agent.set_task(task_id)
    rewards = []
    for i in range(steps):
        obs = env.observe()
        action = agent(obs)
        r = env.pull(action)
        rewards.append(r)

        if (i + 1) % 30 == 0:
            print(f"  Step {i+1:3d} | Avg reward (last 30): {np.mean(rewards[-30:]):+.3f} "
                  f"| Drift score: {agent.perception.drift_score:.3f}")

    phase_mean = np.mean(rewards)
    print(f"  Phase result: mean reward = {phase_mean:+.3f}")
    return rewards


def main():
    print("=" * 60)
    print("  EvoWrap v0.1.0 — Full Integration Demo")
    print("  Agent: epsilon-greedy bandit  |  Arms:", N_ARMS)
    print("=" * 60)

    # Phase 1: Learn the initial distribution
    r1 = run_phase("Initial Learning", 60, "task_A")

    # Phase 2: Environment shifts — agent must detect drift and adapt
    env.shift(delta=3.0)
    r2 = run_phase("After Distribution Shift", 90, "task_B")

    # Phase 3: Shift again — continual adaptation
    env.shift(delta=-2.0)
    r3 = run_phase("Second Shift", 90, "task_C")

    # Phase 4: Return to original-ish distribution — test forgetting
    env.shift(delta=0.0)
    r4 = run_phase("Return to Baseline", 60, "task_A_retest")

    # ===== 5. Final evaluation =============================================
    print("\n" + "=" * 60)
    print("  FINAL STATS")
    print("=" * 60)
    stats = agent.stats
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Run built-in benchmark
    print("\n" + "=" * 60)
    print("  SEQUENTIAL TASK BENCHMARK")
    print("=" * 60)
    bench_results = agent.evaluator.run_benchmark(
        agent_fn=agent._current_fn,
        reward_fn=lambda obs, act: env.pull(act),
        n_arms=N_ARMS,
        n_phases=3,
        steps_per_phase=40,
    )
    for k, v in bench_results.items():
        if isinstance(v, list):
            print(f"  {k}: {[f'{x:.3f}' for x in v]}")
        else:
            print(f"  {k}: {v:.4f}")

    # Print evaluation report
    print("\n" + agent.report())

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    total_reward = np.mean(r1 + r2 + r3 + r4)
    print(f"  Total mean reward across all phases: {total_reward:+.3f}")
    print(f"  Evolution cycles triggered: {agent._evolution_count}")
    print(f"  Memory entries: {len(agent.memory.long_term)}")
    print(f"  Agent genome: {agent.genome.params}")
    print("=" * 60)
    print("  Demo complete.")


if __name__ == "__main__":
    main()
