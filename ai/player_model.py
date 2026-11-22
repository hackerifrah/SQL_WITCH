"""NULL's player model.

NGram: predicts Ada's next technique from her last ones (N = 1, 2, 3 mixed with Witten-Bell
interpolation, Millington ch. 7.3).
CounterBandit: Beta beliefs of how well each counter works against each kind of cast,
picked with Thompson sampling and updated after every exchange.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

PLAIN = "plain"


def signature(runes):
    """('WHERE', 'JOIN') -> 'JOIN+WHERE'"""
    return "+".join(sorted(runes)) or PLAIN


def entropy(dist):
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


@dataclass
class NGram:
    n: int = 3
    counts: dict = field(default_factory=lambda: defaultdict(Counter))
    totals: Counter = field(default_factory=Counter)

    def observe(self, history):
        if not history:
            return
        last = history[-1]
        self.totals[last] += 1
        for k in range(1, self.n):
            if len(history) > k:
                context = tuple(history[-1 - k:-1])
                self.counts[context][last] += 1

    def train(self, sequence):
        for i in range(len(sequence)):
            self.observe(sequence[:i + 1])

    def prob(self, token, context, vocab):
        if not context:
            return (self.totals[token] + 1) / (sum(self.totals.values()) + len(vocab))
        lower = self.prob(token, context[1:], vocab)
        seen = self.counts.get(context)
        if not seen:
            return lower
        total = sum(seen.values())
        trust = total / (total + len(seen))
        return trust * seen[token] / total + (1 - trust) * lower

    def predict(self, history, vocab):
        context = tuple(history[-(self.n - 1):]) if self.n > 1 else ()
        raw = {t: self.prob(t, context, vocab) for t in vocab}
        total = sum(raw.values()) or 1.0
        return {t: v / total for t, v in raw.items()}

    def order_used(self, history):
        for k in range(min(self.n - 1, len(history)), 0, -1):
            if self.counts.get(tuple(history[-k:])):
                return k + 1
        return 1


@dataclass
class CounterBandit:
    priors: dict
    alpha: dict = field(default_factory=dict)
    beta: dict = field(default_factory=dict)
    pulls: Counter = field(default_factory=Counter)

    def __post_init__(self):
        for key, (a, b) in self.priors.items():
            self.alpha.setdefault(key, float(a))
            self.beta.setdefault(key, float(b))

    def sample(self, move, kind, rng):
        return rng.betavariate(self.alpha[(move, kind)], self.beta[(move, kind)])

    def mean(self, move, kind):
        a, b = self.alpha[(move, kind)], self.beta[(move, kind)]
        return a / (a + b)

    def update(self, move, kind, reward):
        reward = min(1.0, max(0.0, reward))
        self.alpha[(move, kind)] += reward
        self.beta[(move, kind)] += 1.0 - reward
        self.pulls[(move, kind)] += 1

    def learned(self, move, kind):
        a0, b0 = self.priors[(move, kind)]
        return self.mean(move, kind) - a0 / (a0 + b0)
