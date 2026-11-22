"""Behaviour tree nodes for Count Aggregor (Millington ch. 5.4).

Each node returns SUCCESS, FAILURE or RUNNING when ticked. The tree is ticked once per turn
from the root, so a higher branch can interrupt a plan that was running.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


MARK = {Status.SUCCESS: "+", Status.FAILURE: "x", Status.RUNNING: "~"}


@dataclass
class Blackboard:
    rng: random.Random
    turn: int = 0
    data: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    depth: int = 0
    cooldowns: dict = field(default_factory=dict)
    uses: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

    def log(self, text):
        self.trace.append("  " * self.depth + text)


class Node:
    name = "node"

    def __init__(self, name=None):
        if name is not None:
            self.name = name

    def tick(self, bb):
        slot = len(bb.trace)
        bb.trace.append("")
        bb.depth += 1
        try:
            status = self.run(bb)
        finally:
            bb.depth -= 1
        bb.trace[slot] = "  " * bb.depth + f"{MARK[status]} {self.label()}"
        return status

    def run(self, bb):
        raise NotImplementedError

    def label(self):
        return self.name


class Composite(Node):
    def __init__(self, name, *children):
        super().__init__(name)
        self.children = list(children)

    def order(self, bb):
        return list(self.children)


class Selector(Composite):
    def run(self, bb):
        for child in self.order(bb):
            match child.tick(bb):
                case Status.SUCCESS:
                    return Status.SUCCESS
                case Status.RUNNING:
                    return Status.RUNNING
        return Status.FAILURE

    def label(self):
        return f"?{self.name}"


class Sequence(Composite):
    def run(self, bb):
        for child in self.order(bb):
            match child.tick(bb):
                case Status.FAILURE:
                    return Status.FAILURE
                case Status.RUNNING:
                    return Status.RUNNING
        return Status.SUCCESS

    def label(self):
        return f">{self.name}"


class RandomSelector(Selector):
    """Tries the children in a random order picked with weights (numbers or functions)."""

    def __init__(self, name, weights, *children):
        super().__init__(name, *children)
        self.weights = list(weights)

    def order(self, bb):
        pool = list(self.children)
        w = [max(0.0, x(bb) if callable(x) else float(x)) for x in self.weights]
        bb.data.setdefault("weights", {})[self.name] = {
            c.name: round(x, 2) for c, x in zip(self.children, w)}
        result = []
        while pool:
            if sum(w) <= 0:
                result += pool
                break
            i = bb.rng.choices(range(len(pool)), w)[0]
            result.append(pool.pop(i))
            w.pop(i)
        return result

    def label(self):
        return f"?~{self.name}"


class UtilitySelector(Selector):
    """Tries the children from best score to worst (scores get a bit of noise)."""

    def __init__(self, name, scores, *children, noise=(0.9, 1.1)):
        super().__init__(name, *children)
        self.scores = scores
        self.noise = noise

    def order(self, bb):
        rated = [(score(bb) * bb.rng.uniform(*self.noise), i) for i, score in enumerate(self.scores)]
        bb.data.setdefault("utilities", {})[self.name] = {
            self.children[i].name: round(s, 2) for s, i in rated}
        rated.sort(key=lambda t: (-t[0], t[1]))
        return [self.children[i] for _, i in rated]

    def label(self):
        return f"?${self.name}"


class Decorator(Node):
    def __init__(self, name, child):
        super().__init__(name)
        self.child = child


class Inverter(Decorator):
    def run(self, bb):
        match self.child.tick(bb):
            case Status.SUCCESS:
                return Status.FAILURE
            case Status.FAILURE:
                return Status.SUCCESS
            case other:
                return other


class Succeeder(Decorator):
    def run(self, bb):
        self.child.tick(bb)
        return Status.SUCCESS


class Chance(Decorator):
    """Coin flip: the child only runs with probability p."""

    def __init__(self, name, p, child):
        super().__init__(name, child)
        self.p = p

    def run(self, bb):
        p = self.p(bb) if callable(self.p) else self.p
        roll = bb.rng.random()
        if roll >= p:
            bb.log(f"  coin {roll:.2f} >= {p:.2f}")
            return Status.FAILURE
        return self.child.tick(bb)

    def label(self):
        return f"{self.name} (chance)"


class Cooldown(Decorator):
    """After the child succeeds it can't run again for some turns. Same key = shared cooldown."""

    def __init__(self, name, turns, child, key=None):
        super().__init__(name, child)
        self.turns = turns
        self.key = key or name

    def run(self, bb):
        ready = bb.cooldowns.get(self.key, 0)
        if bb.turn < ready:
            bb.log(f"  ready on turn {ready}")
            return Status.FAILURE
        status = self.child.tick(bb)
        if status == Status.SUCCESS:
            bb.cooldowns[self.key] = bb.turn + self.turns + 1
        return status


class Limit(Decorator):
    """The child can succeed at most `times` times per battle."""

    def __init__(self, name, times, child):
        super().__init__(name, child)
        self.times = times

    def run(self, bb):
        if bb.uses.get(self.name, 0) >= self.times:
            return Status.FAILURE
        status = self.child.tick(bb)
        if status == Status.SUCCESS:
            bb.uses[self.name] = bb.uses.get(self.name, 0) + 1
        return status


class Condition(Node):
    def __init__(self, name, test):
        super().__init__(name)
        self.test = test

    def run(self, bb):
        return Status.SUCCESS if self.test(bb) else Status.FAILURE

    def label(self):
        return f"{self.name}?"


class Action(Node):
    def __init__(self, name, do):
        super().__init__(name)
        self.do = do

    def run(self, bb):
        result = self.do(bb)
        if isinstance(result, Status):
            return result
        return Status.SUCCESS if result else Status.FAILURE


def success_path(trace):
    """The nodes that really decided the turn (succeeded, and all their parents did too)."""
    path = []
    ok_at = {}
    for line in trace:
        text = line.lstrip()
        if not text or text[0] not in MARK.values():
            continue
        depth = (len(line) - len(text)) // 2
        ok = text[0] in "+~" and all(ok_at.get(d, True) for d in range(depth))
        ok_at[depth] = ok
        for d in [d for d in ok_at if d > depth]:
            del ok_at[d]
        if ok:
            path.append(text[2:])
    return path


def tried_branches(trace, depth=1):
    out = []
    for line in trace:
        text = line.lstrip()
        if text and text[0] in MARK.values() and (len(line) - len(text)) // 2 == depth:
            out.append(text)
    return out


def draw_tree(node, indent=0):
    lines = ["  " * indent + node.label()]
    kids = getattr(node, "children", None) or ([node.child] if hasattr(node, "child") else [])
    for kid in kids:
        lines += draw_tree(kid, indent + 1)
    return lines
