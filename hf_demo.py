#!/usr/bin/env python3
"""
EvoWrap HuggingFace Demo — Colab-Ready
=======================================

Wraps a local HuggingFace model with EvoWrap and runs a multi-phase math
benchmark to exercise the full framework: perception, drift detection,
evolutionary optimization, Three Laws safety, and evaluation.

Designed for Google Colab (free T4 GPU). No API keys needed.

Quick start on Colab:
    !git clone https://github.com/<you>/EvoWrap.git && cd EvoWrap
    !pip install -e . transformers torch accelerate
    !python hf_demo.py

Local usage:
    # Full run with default model (Qwen/Qwen2.5-1.5B-Instruct)
    python hf_demo.py

    # Custom model
    python hf_demo.py --model microsoft/Phi-3-mini-4k-instruct

    # Mock backend (no GPU, no download — tests framework only)
    python hf_demo.py --model mock

Inference budget (~150 total with defaults):
    - 30 direct calls (10 per phase) + 10 retest
    - ~60-90 per evolution cycle (pop=3 x inputs=10 x gens=3)
    - 1-2 evolution cycles expected
    - ~60-90s on Colab T4 at ~0.3s/call
"""

import argparse
import json
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from evowrap import (
    EvoWrapAgent,
    Evaluator,
    Genome,
    OptimizerEngine,
    PerceptionModule,
    SafetyChecker,
)


# ============================================================================
#  HuggingFace Backend
# ============================================================================

class HFBackend:
    """Local HuggingFace model via transformers."""

    name = "huggingface"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs: Dict[str, Any] = {}
        if self.device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        print(f"  Model loaded ({self.device}).")

    def call(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # Use chat template if available, otherwise fall back to manual format
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = f"System: {system}\nUser: {prompt}\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        temp = max(0.01, float(temperature))  # avoid 0 for do_sample
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================================
#  Mock Backend (no model needed)
# ============================================================================

class MockBackend:
    """
    Fake LLM that parses numbers from the question and computes answers.
    Useful for testing the EvoWrap framework without downloading any model.
    Deliberately noisy — gets ~70% right so the optimizer has something to do.
    """

    name = "mock"

    def __init__(self):
        self._rng = np.random.RandomState(123)

    def call(self, prompt, system, temperature, max_tokens):
        numbers = re.findall(r'\d+\.?\d*', prompt)
        nums = [float(n) for n in numbers] if numbers else [0.0]

        p = prompt.lower()
        if any(w in p for w in ["sum", "add", "plus", "total", "+"]):
            result = sum(nums)
        elif any(w in p for w in ["subtract", "minus", "difference", "-"]):
            result = nums[0] - nums[-1] if len(nums) >= 2 else nums[0]
        elif any(w in p for w in ["multiply", "product", "times", "*", "x"]):
            result = np.prod(nums)
        elif any(w in p for w in ["divide", "quotient", "/"]):
            result = nums[0] / nums[-1] if len(nums) >= 2 and nums[-1] != 0 else 0
        elif any(w in p for w in ["percent", "%"]):
            result = nums[0] * nums[-1] / 100 if len(nums) >= 2 else nums[0]
        else:
            result = nums[-1] if nums else 0

        noise = self._rng.normal(0, temperature * 2)
        if self._rng.random() < 0.15 + temperature * 0.1:
            result += noise

        result = round(result, 2)

        if "ANSWER" in system:
            return f"Working... ANSWER: {result}"
        elif "step by step" in system.lower():
            return f"Let me think step by step.\nFirst I see: {nums}\nTherefore, the answer is {result}"
        elif "only" in system.lower() or "just" in system.lower():
            return str(result)
        else:
            return f"The answer is {result}."


# ============================================================================
#  Question Bank — 3 phases x 10 questions (trimmed for speed)
# ============================================================================

PHASE_1 = [
    {"q": "What is 15 + 27?", "a": 42},
    {"q": "What is 8 * 7?", "a": 56},
    {"q": "What is 144 / 12?", "a": 12},
    {"q": "What is 99 - 37?", "a": 62},
    {"q": "What is 25 * 4?", "a": 100},
    {"q": "What is 1000 - 387?", "a": 613},
    {"q": "What is 13 + 28 + 9?", "a": 50},
    {"q": "What is 72 / 8?", "a": 9},
    {"q": "What is 15 * 15?", "a": 225},
    {"q": "What is 500 - 123?", "a": 377},
]

PHASE_2 = [
    {"q": "A store sells apples for $3 each. If you buy 7 apples, how much do you pay in total?", "a": 21},
    {"q": "A train travels at 60 miles per hour for 3 hours. How many miles does it cover?", "a": 180},
    {"q": "If you have 48 cookies and divide them equally among 6 friends, how many cookies does each friend get?", "a": 8},
    {"q": "A rectangle has a length of 12 cm and a width of 5 cm. What is its area in square cm?", "a": 60},
    {"q": "You save $15 per week. How much do you save in 8 weeks?", "a": 120},
    {"q": "A book has 350 pages. If you read 50 pages per day, how many days to finish?", "a": 7},
    {"q": "There are 24 students in a class. 3/4 of them passed the exam. How many passed?", "a": 18},
    {"q": "You bought 3 shirts at $25 each and 2 pants at $40 each. What is the total cost?", "a": 155},
    {"q": "A factory produces 120 units per hour. How many units in a 8 hour shift?", "a": 960},
    {"q": "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?", "a": 2},
]

PHASE_3 = [
    {"q": "What is the next number in the sequence: 2, 6, 18, 54, ?", "a": 162},
    {"q": "If a square has a perimeter of 36 cm, what is the length of one side in cm?", "a": 9},
    {"q": "What is 2 to the power of 10?", "a": 1024},
    {"q": "How many seconds are in 2 hours?", "a": 7200},
    {"q": "What is 15 percent of 200?", "a": 30},
    {"q": "A number doubled and then increased by 10 gives 50. What is the number?", "a": 20},
    {"q": "What is the average of 10, 20, 30, 40, and 50?", "a": 30},
    {"q": "If x + 5 = 12, what is x * 3?", "a": 21},
    {"q": "How many minutes are in 3.5 hours?", "a": 210},
    {"q": "What is the next number in the sequence: 1, 4, 9, 16, 25, ?", "a": 36},
]


# ============================================================================
#  Answer Extraction & Reward
# ============================================================================

def _parse_number(s: str) -> Optional[float]:
    """Parse a number string, stripping commas."""
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def extract_number(text: str) -> Optional[float]:
    """Extract the numerical answer from LLM output."""
    if not text or not text.strip():
        return None

    # Priority 1: Explicit answer markers
    answer_patterns = [
        r'(?:ANSWER|Answer|answer)\s*[:=]\s*\$?\s*([+-]?\d[\d,]*\.?\d*)',
        r'\\boxed\{([+-]?\d[\d,]*\.?\d*)\}',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            val = _parse_number(match.group(1))
            if val is not None:
                return val

    # Priority 2: "Final answer line" heuristic — check lines starting with
    # answer-indicating words
    lines = text.strip().split("\n")
    for line in lines:
        stripped = line.strip()
        if re.match(r'^(?:Answer|ANSWER|The answer|Therefore|So|Thus|Hence)\b', stripped, re.IGNORECASE):
            nums = re.findall(r'([+-]?\d[\d,]*\.?\d*)', stripped)
            if nums:
                val = _parse_number(nums[-1])
                if val is not None:
                    return val

    # Priority 3: Common phrasing patterns
    phrase_patterns = [
        r'(?:answer is|result is|equals|total is|gives|value is)\s*\$?\s*([+-]?\d[\d,]*\.?\d*)',
        r'(?:therefore|so|thus|hence)[,\s]+(?:the\s+)?(?:answer|result|total|value)\s+is\s*\$?\s*([+-]?\d[\d,]*\.?\d*)',
        r'(\d[\d,]*\.?\d*)\s*(?:is the answer|is the result|is the total)',
        r'\*\*\$?\s*([+-]?\d[\d,]*\.?\d*)\s*\*\*',
        r'=\s*\$?\s*([+-]?\d[\d,]*\.?\d*)\s*(?:\$)?\s*$',
    ]
    for pattern in phrase_patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            val = _parse_number(match.group(1))
            if val is not None:
                return val

    # Priority 4: Number before common unit words
    unit_pattern = r'(\d[\d,]*\.?\d*)\s*(?:days?|hours?|minutes?|seconds?|miles?|cm|meters?|units?|slices?|cookies?|students?|dollars?|\$)'
    match = re.search(unit_pattern, text, re.IGNORECASE)
    if match:
        val = _parse_number(match.group(1))
        if val is not None:
            return val

    # Priority 5: Last number in the last 3 lines (where answers usually are)
    last_lines = "\n".join(lines[-3:]) if len(lines) >= 3 else text
    numbers = re.findall(r'(?<![a-zA-Z])([+-]?\d[\d,]*\.?\d*)(?![a-zA-Z])', last_lines)
    if numbers:
        for n in reversed(numbers):
            val = _parse_number(n)
            if val is not None:
                return val

    # Priority 6: Last number in entire text (ultimate fallback)
    numbers = re.findall(r'(?<![a-zA-Z])([+-]?\d[\d,]*\.?\d*)(?![a-zA-Z])', text)
    if numbers:
        for n in reversed(numbers):
            val = _parse_number(n)
            if val is not None:
                return val

    return None


def make_reward_fn(questions: List[Dict]) -> Callable:
    """Create a reward function that scores LLM output against ground truth."""
    lookup = {q["q"]: q["a"] for q in questions}

    def reward_fn(question: str, response: str) -> float:
        expected = lookup.get(question)
        if expected is None:
            return 0.0
        extracted = extract_number(str(response))
        if extracted is None:
            return 0.0
        if abs(extracted - expected) < 0.5:
            return 1.0
        if expected != 0:
            relative_error = abs(extracted - expected) / abs(expected)
            if relative_error < 0.05:
                return 0.8
            if relative_error < 0.15:
                return 0.4
        return 0.0

    return reward_fn


# ============================================================================
#  Genome, Build Function
# ============================================================================

_backend = None
_api_call_count = 0


def build_agent(genome: Genome) -> Callable:
    """Build an LLM agent function from evolved genome parameters."""
    temp = max(0.0, min(1.5, float(genome.params.get("temperature", 0.3))))
    max_tok = max(30, min(300, int(genome.params.get("max_tokens", 100))))
    system_prompt = str(genome.params.get(
        "system_prompt",
        "You are a precise math solver. Think step by step, then write your final answer on the last line as: ANSWER: <number>",
    ))
    output_format = str(genome.params.get("output_format", "End your response with exactly: ANSWER: <number>"))
    full_system = f"{system_prompt}\n{output_format}"

    def agent(question: str) -> str:
        global _api_call_count
        _api_call_count += 1
        return _backend.call(question, full_system, temp, max_tok)

    return agent


# ============================================================================
#  Run Phase
# ============================================================================

def run_phase(
    agent: EvoWrapAgent,
    phase_name: str,
    questions: List[Dict],
    task_id: str,
    reward_fn: Callable,
    verbose: bool = True,
) -> List[float]:
    """Run one phase of the benchmark."""
    print(f"\n{'=' * 65}")
    print(f"  PHASE: {phase_name}  |  {len(questions)} questions  |  task={task_id}")
    print(f"{'=' * 65}")

    agent.set_task(task_id)
    rewards = []
    correct = 0

    for i, item in enumerate(questions):
        q, expected = item["q"], item["a"]
        response = agent(q)

        extracted = extract_number(str(response))
        r = reward_fn(q, str(response))
        rewards.append(r)
        is_correct = r >= 0.8

        if is_correct:
            correct += 1

        if verbose:
            status = "OK" if is_correct else "MISS"
            extracted_str = f"{extracted}" if extracted is not None else "None"
            resp_preview = str(response).replace('\n', ' ')[:60]
            print(
                f"  [{status:4s}] Q{i+1:2d}: expected={expected}, "
                f"got={extracted_str:>8s}  r={r:.1f}  "
                f"| {resp_preview}"
            )

    accuracy = correct / len(questions) if questions else 0
    mean_r = np.mean(rewards) if rewards else 0
    print(f"  -------")
    print(f"  Phase result: {correct}/{len(questions)} correct ({accuracy:.0%}), mean reward={mean_r:.3f}")
    print(f"  Drift score: {agent.perception.drift_score:.3f}")
    return rewards


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoWrap HuggingFace Demo — run on Colab or locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hf_demo.py                                         # default: Qwen2.5-1.5B-Instruct
  python hf_demo.py --model microsoft/Phi-3-mini-4k-instruct
  python hf_demo.py --model mock                            # framework test, no download
        """,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name, or 'mock' for no-download testing (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument("--evolve-every", type=int, default=15, help="Steps between evolution checks")
    parser.add_argument("--population-size", type=int, default=3, help="Optimizer population size")
    parser.add_argument("--max-generations", type=int, default=3, help="Max generations per evolution cycle")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--save-genome", default=None, help="Path to save evolved genome JSON")
    args = parser.parse_args()

    use_mock = args.model.lower() == "mock"

    print("=" * 65)
    print("  EvoWrap HuggingFace Demo")
    print(f"  Model: {'mock (no download)' if use_mock else args.model}")
    print("=" * 65)

    global _backend
    if use_mock:
        _backend = MockBackend()
    else:
        _backend = HFBackend(args.model)

    all_questions = PHASE_1 + PHASE_2 + PHASE_3
    reward_fn = make_reward_fn(all_questions)

    initial_genome = {
        "temperature": 0.3,
        "max_tokens": 100,
        "system_prompt": "You are a precise math solver. Think step by step, then write your final answer on the last line as: ANSWER: <number>",
        "output_format": "End your response with exactly: ANSWER: <number>",
    }

    safety = SafetyChecker(
        perf_tolerance=0.20,
        min_improvement=0.005,
        crash_sim_trials=3,
        max_error_rate=0.35,
    )

    agent = EvoWrapAgent(
        agent_fn=build_agent(Genome(params=initial_genome)),
        genome=initial_genome,
        build_fn=build_agent,
        reward_fn=reward_fn,
        evolve_every=args.evolve_every,
        auto_evolve=True,
        safety=safety,
        optimizer=OptimizerEngine(
            safety=safety,
            population_size=args.population_size,
            elite_fraction=0.25,
            mutation_rate=0.5,
            evolution_trigger_threshold=0.6,
        ),
        perception=PerceptionModule(drift_threshold=0.3),
    )

    verbose = not args.quiet
    t_start = time.time()

    r1 = run_phase(agent, "Arithmetic", PHASE_1, "arithmetic", reward_fn, verbose)
    r2 = run_phase(agent, "Word Problems", PHASE_2, "word_problems", reward_fn, verbose)
    r3 = run_phase(agent, "Logic & Sequences", PHASE_3, "logic", reward_fn, verbose)

    # Phase 4: Forgetting retest — rerun Phase 1 questions
    print(f"\n{'=' * 65}")
    print(f"  PHASE: Retest Arithmetic (forgetting measurement)")
    print(f"{'=' * 65}")
    agent.set_task("arithmetic_retest")
    retest_rewards = []
    retest_correct = 0
    for i, item in enumerate(PHASE_1):
        q, expected = item["q"], item["a"]
        response = agent(q)
        extracted = extract_number(str(response))
        r = reward_fn(q, str(response))
        retest_rewards.append(r)
        is_correct = r >= 0.8
        if is_correct:
            retest_correct += 1
        if verbose:
            status = "OK" if is_correct else "MISS"
            extracted_str = f"{extracted}" if extracted is not None else "None"
            resp_preview = str(response).replace('\n', ' ')[:60]
            print(
                f"  [{status:4s}] Q{i+1:2d}: expected={expected}, "
                f"got={extracted_str:>8s}  r={r:.1f}  "
                f"| {resp_preview}"
            )

    p1_accuracy = sum(1 for r in r1 if r >= 0.8) / len(r1)
    retest_accuracy = retest_correct / len(PHASE_1)
    forgetting_delta = retest_accuracy - p1_accuracy
    print(f"  -------")
    print(f"  Retest result: {retest_correct}/{len(PHASE_1)} correct ({retest_accuracy:.0%})")
    print(f"  Phase 1 original: {p1_accuracy:.0%} → Retest: {retest_accuracy:.0%}")
    print(f"  Forgetting delta: {forgetting_delta:+.0%} {'(no forgetting)' if forgetting_delta >= 0 else '(forgetting detected)'}")

    elapsed = time.time() - t_start

    # --- Final report ---
    print(f"\n{'=' * 65}")
    print("  FINAL RESULTS")
    print(f"{'=' * 65}")

    stats = agent.stats
    print(f"  Total steps:         {stats['step']}")
    print(f"  Evolution cycles:    {stats['evolution_cycles']}")
    print(f"  Overall mean reward: {stats['mean_reward']:.3f}")
    print(f"  Success rate:        {stats['success_rate']:.1%}")
    print(f"  Final drift score:   {stats['drift_score']:.3f}")
    print(f"  Memory entries:      {stats['memory']['long_term_size']}")
    print(f"  Total inferences:    {_api_call_count}")
    print(f"  Wall time:           {elapsed:.1f}s")

    print(f"\n  Initial genome: temperature={initial_genome['temperature']}, max_tokens={initial_genome['max_tokens']}")
    print(f"  Evolved genome: temperature={agent.genome.params.get('temperature', '?')}, max_tokens={agent.genome.params.get('max_tokens', '?')}")
    sys_prompt_preview = str(agent.genome.params.get('system_prompt', ''))[:70]
    print(f"  System prompt:  \"{sys_prompt_preview}...\"")

    print(f"\n  Per-phase accuracy:")
    for name, rewards in [("Arithmetic", r1), ("Word Problems", r2), ("Logic", r3), ("Retest Arithmetic", retest_rewards)]:
        acc = sum(1 for r in rewards if r >= 0.8) / len(rewards)
        print(f"    {name:20s}: {acc:.0%} ({sum(1 for r in rewards if r >= 0.8)}/{len(rewards)})")

    if agent.optimizer.evolution_history:
        print(f"\n  Evolution history:")
        for entry in agent.optimizer.evolution_history:
            print(f"    {entry}")

    print(f"\n{agent.report()}")

    if args.save_genome:
        genome_data = {
            "params": agent.genome.params,
            "fitness": agent.genome.fitness,
            "generation": agent.genome.generation,
            "lineage": agent.genome.lineage,
            "model": args.model,
            "stats": {
                "total_steps": stats["step"],
                "mean_reward": stats["mean_reward"],
                "inferences": _api_call_count,
            },
        }
        with open(args.save_genome, "w") as f:
            json.dump(genome_data, f, indent=2, default=str)
        print(f"\n  Genome saved to {args.save_genome}")

    print(f"\n{'=' * 65}")
    print("  Demo complete.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
