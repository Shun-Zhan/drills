"""Feasibility-first reward with exact arithmetic for its audit."""
from fractions import Fraction

from drills.fpga_session import FPGASession

from context import SPEC


class RewardTracker:
    def __init__(self, initial_luts, initial_levels, max_levels):
        self.initial_luts = int(initial_luts)
        self.initial_levels = int(initial_levels)
        self.max_levels = int(max_levels)
        self.seen_feasible = self.initial_levels <= self.max_levels
        self.best_luts = self.initial_luts if self.seen_feasible else None
        self.lowest_levels = self.initial_levels
        self.depth_bonus = Fraction(0)

    def step_exact(self, luts, levels):
        luts, levels = int(luts), int(levels)
        scale = SPEC['reward']
        if levels <= self.max_levels:
            if not self.seen_feasible:
                value = (Fraction(scale['first_feasible']) +
                         Fraction(scale['lut_percent'] * (self.initial_luts - luts),
                                  self.initial_luts) - self.depth_bonus)
                self.seen_feasible = True
                self.best_luts = luts
                return value
            if luts < self.best_luts:
                value = Fraction(scale['lut_percent'] * (self.best_luts - luts),
                                 self.initial_luts)
                self.best_luts = luts
                return value
            return Fraction(0)
        if not self.seen_feasible and levels < self.lowest_levels:
            value = Fraction(scale['infeasible_depth'] * (self.lowest_levels - levels),
                             max(self.initial_levels - self.max_levels, 1))
            self.depth_bonus += value
            self.lowest_levels = levels
            return value
        return Fraction(0)

    def step(self, luts, levels):
        return float(self.step_exact(luts, levels))


def original_reward(cfg, circuit, previous, current):
    before_luts, before_levels = previous
    luts, levels = current
    area = int(luts < before_luts) - int(luts > before_luts)
    if levels <= circuit['max_levels']:
        return cfg['method']['reward']['feasible'][area]
    depth = int(levels < before_levels) - int(levels > before_levels)
    return cfg['method']['reward']['infeasible'][depth][area]


class RewardSession(FPGASession):
    """The original ABC environment with only _get_reward changed."""
    def reset(self):
        state = super().reset()
        self.reward_tracker = RewardTracker(self.luts, self.levels, self.circuit['max_levels'])
        return state

    def _get_reward(self, luts, levels):
        return self.reward_tracker.step(luts, levels)
