"""Barks: short lines a guardian says, picked by rules about what it just perceived
(like the bark systems in F.E.A.R. and Halo).

Every rule has a condition, some lines and a priority. The highest priority rule whose
condition is true and that isn't cooling down speaks. The same line is never said twice in a row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

COOLDOWN = 3
MAX_LEN = 44


@dataclass
class Bark:
    name: str
    when: object
    lines: tuple
    priority: int = 1


@dataclass
class Barker:
    rules: list
    rest_until: dict = field(default_factory=dict)
    last_line: str | None = None
    said: list = field(default_factory=list)

    def speak(self, info, turn, rng):
        ready = [b for b in self.rules if self.rest_until.get(b.name, 0) <= turn and check(b, info)]
        if not ready:
            return None
        top = max(b.priority for b in ready)
        rule = rng.choice([b for b in ready if b.priority == top])
        options = [line for line in rule.lines if line != self.last_line] or list(rule.lines)
        line = rng.choice(options).format(**info)[:MAX_LEN]
        self.rest_until[rule.name] = turn + COOLDOWN
        self.last_line = line
        self.said.append(line)
        return line


def check(bark, info):
    try:
        return bool(bark.when(info))
    except (KeyError, TypeError, ZeroDivisionError):
        return False
