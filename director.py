"""AI Director (dynamic difficulty, like Left 4 Dead's director).

Each turn it compares Ada's HP share with the guardian's and keeps a moving average.
intensity > 1: Ada is winning, so the guardian plays sharper.
intensity < 1: Ada is struggling, so the guardian eases off a bit.
It only changes how the guardian decides, never damage, HP or MP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SMOOTHING = 0.35
LOWEST, HIGHEST = 0.75, 1.3
GAIN = 0.6


@dataclass
class Director:
    advantage: float = 0.0
    intensity: float = 1.0
    history: list = field(default_factory=list)

    def update(self, ada_hp, my_hp):
        reading = ada_hp - my_hp
        self.advantage = SMOOTHING * reading + (1 - SMOOTHING) * self.advantage
        self.intensity = round(min(HIGHEST, max(LOWEST, 1.0 + GAIN * self.advantage)), 3)
        self.history.append(self.intensity)
        return self.intensity

    @property
    def mood(self):
        if self.intensity > 1.08:
            return "sharper (Ada leads)"
        if self.intensity < 0.92:
            return "easing off (Ada struggles)"
        return "steady"

    def line(self):
        return f"{self.mood}, intensity {self.intensity:.2f} (advantage {self.advantage:+.2f})"
