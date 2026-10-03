"""Portable orchestration settings; environment adapters own action conventions."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AcroConfig:
    max_steps: int
    history_queries: int = 8
    min_target_gap: int = 20
    lookback_steps: tuple[int | None, ...] = (120, 300, None)
    max_interventions: int | None = 3
    max_retraction_steps: int | None = 200
    tube_radius: float = 0.02
    tube_sample_spacing: float = 0.005
    position_tolerance: float = 0.005
    rotation_tolerance: float = 0.04

    def __post_init__(self):
        counts = (self.max_steps, self.history_queries, self.min_target_gap)
        if any(type(n) is not int or n < 1 for n in counts):
            raise ValueError("step and history limits must be positive integers")
        if (self.max_interventions is not None and
                (type(self.max_interventions) is not int or self.max_interventions < 0)):
            raise ValueError("intervention limit must be nonnegative or None")
        if (self.max_retraction_steps is not None and
                (type(self.max_retraction_steps) is not int or self.max_retraction_steps < 1)):
            raise ValueError("retraction limit must be positive or None")
        if not self.lookback_steps or any(
            n is not None and (type(n) is not int or n < self.min_target_gap)
            for n in self.lookback_steps
        ):
            raise ValueError("look-back windows must cover the minimum target gap")
        tolerances = (self.tube_radius, self.tube_sample_spacing,
                      self.position_tolerance, self.rotation_tolerance)
        if any(not math.isfinite(x) or x <= 0 for x in tolerances):
            raise ValueError("path tolerances must be finite and positive")
