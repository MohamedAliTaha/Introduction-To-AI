"""Vacuum cleaner world simulation (Reflex vs. Model-based agents).

Intro to AI Section Task:
Simulates a grid/room environment with stochastic events (dirt reappearing, suck failure)
and compares a Simple Reflex Agent against a Model-Based Reflex Agent.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

Percept = tuple[str, str]  # (location, status)
AgentFn = Callable[[Percept], str]


class VacuumEnv:
    """Vacuum environment consisting of discrete rooms with dirt status."""

    def __init__(
        self,
        rooms: Sequence[str] = ("A", "B"),
        dirty: Sequence[str] = ("A", "B"),
        start: str = "A",
        p_dirty_again: float = 0.0,
        p_suck_fail: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.rooms = list(rooms)
        self.status = {r: ("Dirty" if r in dirty else "Clean") for r in self.rooms}
        self.loc = start
        self.p_dirty_again = p_dirty_again  # احتمال إن غرفة نضيفة تتوسخ تاني
        self.p_suck_fail = p_suck_fail      # احتمال إن Suck يفشل
        self.rng = random.Random(seed)

    def percept(self) -> Percept:
        """Returns the current percept for the agent: (location, status)."""
        return (self.loc, self.status[self.loc])

    def step(self, action: str) -> float:
        """Applies an action to the environment and returns the reward.

        Actions:
            - 'Suck': Cleans current room (may fail with probability p_suck_fail).
            - 'Right': Moves one room to the right if not at boundary.
            - 'Left': Moves one room to the left if not at boundary.
            - 'NoOp': Does nothing.
        """
        i = self.rooms.index(self.loc)
        move_cost = 0.0

        if action == "Suck":
            if self.rng.random() >= self.p_suck_fail:
                self.status[self.loc] = "Clean"
        elif action == "Right" and i < len(self.rooms) - 1:
            self.loc = self.rooms[i + 1]
            move_cost = 0.5
        elif action == "Left" and i > 0:
            self.loc = self.rooms[i - 1]
            move_cost = 0.5
        # NoOp: مفيش حاجة تحصل

        # Stochastic dirtying: clean rooms may become dirty again
        for r in self.rooms:
            if self.status[r] == "Clean" and self.rng.random() < self.p_dirty_again:
                self.status[r] = "Dirty"

        clean_rooms = sum(s == "Clean" for s in self.status.values())
        return clean_rooms - move_cost


def reflex_vacuum(percept: Percept) -> str:
    """Simple Reflex Agent:

    Acts only on the current percept without internal memory.
    """
    location, status = percept
    if status == "Dirty":
        return "Suck"
    elif location == "A":
        return "Right"
    else:
        return "Left"


class ModelBasedVacuum:
    """Model-based Reflex Agent:

    Maintains internal state of which rooms are known to be clean.
    """

    def __init__(self, rooms: Sequence[str] = ("A", "B")) -> None:
        self.rooms = list(rooms)
        self.known_clean: set[str] = set()

    def __call__(self, percept: Percept) -> str:
        location, status = percept

        if status == "Dirty":
            self.known_clean.discard(location)
            return "Suck"

        self.known_clean.add(location)

        # Stop moving if all rooms are believed clean
        if len(self.known_clean) == len(self.rooms):
            return "NoOp"

        return "Right" if location == "A" else "Left"


def run(
    agent: AgentFn,
    env: VacuumEnv,
    steps: int = 20,
    verbose: bool = False,
) -> float:
    """Runs an agent in an environment for a fixed number of steps."""
    total_reward = 0.0
    for t in range(steps):
        p = env.percept()
        a = agent(p)
        r = env.step(a)
        total_reward += r
        if verbose:
            print(f"t={t:2d} | percept={str(p):16} | action={a:6} | reward={r:4.1f} | status={env.status}")
    return total_reward


def evaluate(
    agent_factory: Callable[[], AgentFn],
    steps: int = 50,
    runs: int = 200,
    **env_kw,
) -> float:
    """Evaluates average agent performance across multiple random seeds."""
    total = sum(
        run(agent_factory(), VacuumEnv(seed=s, **env_kw), steps=steps)
        for s in range(runs)
    )
    return total / runs


def main() -> None:
    # 1. Step-by-step Trace Demos
    print("=" * 60)
    print("=== Reflex Agent (Trace Demo) ===")
    print("=" * 60)
    reflex_score = run(reflex_vacuum, VacuumEnv(), steps=10, verbose=True)
    print(f"Final Score: {reflex_score:.1f}\n")

    print("=" * 60)
    print("=== Model-Based Agent (Trace Demo) ===")
    print("=" * 60)
    model_score = run(ModelBasedVacuum(), VacuumEnv(), steps=10, verbose=True)
    print(f"Final Score: {model_score:.1f}\n")

    # 2. Performance Comparison Across Settings
    settings: dict[str, dict] = {
        "Deterministic": {},
        "Dirt comes back (p=0.10)": {"p_dirty_again": 0.10},
        "Suck fails (p=0.30)": {"p_suck_fail": 0.30},
    }

    print("=" * 60)
    print(f"{'Environment':<30}{'Reflex':>12}{'Model-based':>16}")
    print("-" * 60)
    for name, kw in settings.items():
        r = evaluate(lambda: reflex_vacuum, **kw)
        m = evaluate(lambda: ModelBasedVacuum(), **kw)
        print(f"{name:<30}{r:12.2f}{m:16.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
