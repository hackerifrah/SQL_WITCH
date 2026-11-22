"""NULL, the Voice of Doubt: a guardian that learns how Ada casts and counters it.

Turn: learn from the last exchange (N-gram + bandit) -> predict Ada's next technique ->
stance (match/case) -> look ahead (simulate the next exchange for every move) ->
utility = lookahead + Thompson samples (the AI Director sets how much it explores) ->
Think runs the best plan -> Intent + movement -> a bark about what it predicted.
"""

from __future__ import annotations

import random
from collections import Counter

from ai.barks import Bark, Barker
from ai.base import Intent
from ai.director import Director
from ai.goals import AtomicGoal, CompositeGoal, Context, Evaluator
from ai.guardian import REST, GuardianBrain
from ai.perception import summary
from ai.player_model import CounterBandit, NGram, entropy, signature
from ai.steering import Movement

NULL_ID = 20
VEIL = frozenset({"IS NULL", "COALESCE"})
KINDS = ("blind", "repeat", "fixed", "adaptive")
MOVES = ("strike", "echo", "doubt", "shift", "fade")

ACTION_SPELLS = {
    "strike": "Empty Strike", "echo": "Echo of Doubt", "doubt": "Whisper of Doubt",
    "shift": "Shift of Nothing", "fade": "Fade to Grey", "heal": "Hollow Mend",
}
DEFAULT_COSTS = {"strike": 5, "echo": 7, "doubt": 5, "shift": 6, "fade": 6, "heal": 9}
NULL_INTENTS = {
    "strike": Intent("strike", "Ada", "attack"),
    "echo": Intent("echo", "Ada", "echo"),
    "doubt": Intent("doubt", "Ada", "doubt"),
    "shift": Intent("shift", None, "shift"),
    "fade": Intent("fade", None, "fade"),
    "heal": Intent("heal", None, "heal"),
    "rest": Intent("rest", None, "scry"),
}

# my starting guesses: how good is each move against each kind of cast (Beta alpha, beta)
PRIORS = {
    ("strike", "blind"): (4, 2), ("strike", "repeat"): (2, 2),
    ("strike", "fixed"): (2, 2), ("strike", "adaptive"): (2, 2),
    ("echo", "blind"): (2, 2), ("echo", "repeat"): (3, 2),
    ("echo", "fixed"): (3, 2), ("echo", "adaptive"): (3, 2),
    ("doubt", "blind"): (3, 2), ("doubt", "repeat"): (2, 2),
    ("doubt", "fixed"): (2, 2), ("doubt", "adaptive"): (3, 2),
    ("shift", "blind"): (1, 3), ("shift", "repeat"): (2, 2),
    ("shift", "fixed"): (4, 1), ("shift", "adaptive"): (1, 3),
    ("fade", "blind"): (1, 3), ("fade", "repeat"): (5, 1),
    ("fade", "fixed"): (2, 2), ("fade", "adaptive"): (2, 2),
}
KNOWN = ["LIMIT+WHERE", "IS NULL+JOIN+WHERE", "COALESCE+IS NULL+JOIN+WHERE"]

MODE_BIAS = {
    "listening":   {"doubt": 1.5, "strike": 0.7, "fade": 0.6, "shift": 0.6},
    "mirroring":   {"fade": 1.3, "shift": 1.3, "echo": 1.1},
    "doubting":    {},
    "hunting":     {"strike": 1.4, "echo": 1.3, "fade": 0.7, "shift": 0.7},
    "unravelling": {"heal": 1.5, "fade": 1.2, "doubt": 1.1},
}
MOVEMENT = {
    "strike": Movement("pursue", "ada"), "echo": Movement("arrive", "front"),
    "doubt": Movement("arrive", "front"), "shift": Movement("evade", "ada"),
    "fade": Movement("evade", "ada"), "heal": Movement("arrive", "home"),
    "rest": Movement("wander", "home"),
}

LISTEN_UNTIL = 2
CONFIDENT = 0.5
GUESS_HIT = 60.0
STARVE_MP = 16
SIMS = 24
FADE_HORIZON = 3
MODEL_WEIGHT = 0.55
STRIKE_POWER, ECHO_SHARE, ECHO_CAP, DRAIN, MEND = 34, 0.6, 45, 20, 25

TELL_WORDS = ["COALESCE", "IS NULL", "GROUP BY", "DISTINCT", "JOIN", "LIMIT", "WHERE"]

NULL_BARKS = [
    Bark("unravel", lambda i: i["my_hp"] < 0.3,
         ("I... have a value?", "Stop giving me a value!"), 7),
    Bark("predict", lambda i: i["mode"] == "mirroring" and i["word"] and i["move"] in ("fade", "shift", "doubt"),
         ("You will {word} again. I know it.", "{word}, {word}, {word}. Predictable.",
          "I already know your next {word}."), 6),
    Bark("called it", lambda i: i["right"] is True and i["hits"] >= 2,
         ("I knew you'd do that.", "Called it. {hits} out of {guesses}.", "You're so easy to read."), 5),
    Bark("blind", lambda i: i["blind"] >= 0.5,
         ("You still can't see me, can you?", "= NULL is never true, Ada.",
          "Look at the void. IS it there?"), 5),
    Bark("surprised", lambda i: i["right"] is False and i["guesses"] >= 3,
         ("...I didn't see that coming.", "You changed. Interesting.", "Unpredictable. I hate that."), 4),
    Bark("hunting", lambda i: i["ada_hp"] < 0.35,
         ("Your queries were always pointless.", "Nobody reads your dashboards."), 3),
    Bark("listening", lambda i: i["mode"] == "listening",
         ("Show me how you cast, Ada.", "I'm listening. Go on, cast."), 2),
    Bark("move", lambda i: True,
         ("Nothing hits harder than nothing.", "Are you sure you have the mana?",
          "Your own spell, Ada. Feel it.", "I've been watching since the Glade."), 1),
]


def null_legal_intents(state):
    return list(NULL_INTENTS.values())


def classify(token, last_hit):
    runes = set(token.split("+"))
    if not VEIL & runes:
        return "blind"
    if last_hit is not None and token == last_hit:
        return "repeat"
    if "COALESCE" in runes:
        return "adaptive"
    return "fixed"


def make_plan(move):
    return CompositeGoal(move, [
        AtomicGoal("drift", movement=MOVEMENT[move]),
        AtomicGoal(move, NULL_INTENTS[move], MOVEMENT[move],
                   requires=lambda p, m=move: m == "rest" or p.can(m)),
    ])


class NullBrain(GuardianBrain):
    technique = "N-gram prediction + lookahead + Thompson sampling"
    start_mode = "listening"
    repeat_penalty = 0.8

    def __init__(self, costs=None, learning=True, n=3, lookahead=True):
        self.utility = {}
        evaluators = [Evaluator(m, self.desire(m), lambda p, m=m: make_plan(m))
                      for m in (*MOVES, "heal")]
        evaluators.append(Evaluator("rest", lambda p, mode: 0.02, lambda p: make_plan("rest")))
        super().__init__(costs or dict(DEFAULT_COSTS), evaluators)
        self.learning = learning
        self.use_lookahead = lookahead
        self.model = NGram(n)
        self.bandit = CounterBandit(PRIORS)
        self.old_history = []
        self.history = []
        self.seen = 0
        self.last_move = None
        self.last_ada_hp = None
        self.expected_hit = GUESS_HIT
        self.prediction = {}
        self.kind_odds = {}
        self.top_guess = None
        self.last_hit = None
        self.hits = 0
        self.guesses = 0
        self.guess_log = []
        self.harmless = 0.0
        self.free_defence = False
        self.sim_values = {}
        self.last_right = None
        self.turns = 0
        self.director = Director()
        self.barker = Barker(NULL_BARKS)
        self.says = None

    def observe_run(self, history):
        tokens = [h if isinstance(h, str) else signature(h) for h in history]
        self.old_history += tokens
        self.model.train(tokens)

    def desire(self, move):
        return lambda p, mode: self.utility.get(move, 0.0)

    # ------------------------------------------------------------ learning

    def learn(self, state):
        start = self.seen
        self.seen = len(state.ada_casts)
        notes = []
        for i in range(start, len(state.ada_casts)):
            cast = state.ada_casts[i]
            if cast.scry:
                continue
            token = signature(cast.runes)
            if self.top_guess is not None:
                self.guesses += 1
                right = token == self.top_guess
                self.last_right = right
                self.hits += right
                self.guess_log.append({"turn": cast.turn, "predicted": self.top_guess,
                                       "actual": token, "correct": right,
                                       "p_actual": round(self.prediction.get(token, 0.0), 3)})
                self.top_guess = None
            kind = classify(token, find_last_hit(state.ada_casts[:i]))
            if self.last_move in MOVES and self.learning:
                reward = self.reward(cast.damage, state)
                self.bandit.update(self.last_move, kind, reward)
                notes.append(f"{self.last_move} vs {kind}: reward {reward:.2f}")
            if cast.damage > 0:
                self.expected_hit = 0.7 * self.expected_hit + 0.3 * cast.damage
            self.history.append(token)
            if self.learning:
                self.model.observe(self.old_history + self.history)
        return notes

    def reward(self, taken, state):
        """Defensive moves get credit for damage she didn't do, attacks for damage they did."""
        dealt = max(0, (self.last_ada_hp or state.ada.hp) - state.ada.hp)
        attack = min(1.0, dealt / 30)
        if self.last_move in ("strike", "echo"):
            return attack
        defence = 1.0 - min(1.0, taken / max(1.0, self.expected_hit))
        if self.free_defence:
            defence *= 0.3
        return 0.8 * defence + 0.2 * attack

    # ------------------------------------------------------------ predicting

    def predict(self, state):
        history = self.old_history + self.history
        vocab = sorted(set(history) | set(KNOWN))
        self.prediction = self.model.predict(history, vocab)
        self.top_guess = max(self.prediction, key=self.prediction.get)
        self.last_hit = find_last_hit(state.ada_casts)
        odds = Counter()
        for token, chance in self.prediction.items():
            odds[classify(token, self.last_hit)] += chance
        self.kind_odds = {k: round(odds.get(k, 0.0), 3) for k in KINDS}

    def is_faded(self, state):
        fade = state.foe.status.get("fade")
        return fade is not None and self.last_hit == "+".join(fade)

    # ------------------------------------------------------------ looking ahead

    def lookahead(self, p, state, rng):
        """Play the next exchange SIMS times for each move: Ada's cast is drawn from my
        prediction, damage rolls are random. Score = her HP lost - my HP lost (as shares)."""
        sim = random.Random(rng.random())
        kinds = list(self.kind_odds)
        weights = [self.kind_odds[k] for k in kinds]
        if sum(weights) <= 0:
            weights = [1] * len(kinds)
        ada, me = state.ada, state.foe
        real = [c for c in state.ada_casts if not c.scry]
        last_damage = real[-1].damage if real else 0
        faded = self.is_faded(state)
        values = {}
        for move in (*MOVES, "heal"):
            if move != "heal" and not p.can(move):
                continue
            total = 0.0
            for _ in range(SIMS):
                kind = sim.choices(kinds, weights)[0]
                roll = sim.uniform(0.85, 1.0) * (1.5 if sim.random() < 1 / 16 else 1.0)
                dealt = 0.0
                if move == "strike":
                    dealt = STRIKE_POWER * roll
                elif move == "echo":
                    dealt = min(ECHO_CAP, ECHO_SHARE * last_damage) * roll
                taken = self.expected_hit * sim.uniform(0.85, 1.0)
                saved_later = 0.0
                match kind:
                    case "blind":
                        taken = 0
                    case "repeat" if faded:
                        taken = 0
                    case "repeat" if move == "fade":
                        saved_later = taken * FADE_HORIZON * self.kind_odds.get("repeat", 0)
                        taken = 0
                    case "fixed" if move == "shift":
                        taken *= 0.25
                if move == "doubt" and p.ada_mp - DRAIN < 12:
                    taken *= 0.5
                if move == "heal":
                    taken -= MEND
                score = dealt / ada.max_hp - taken / me.max_hp + saved_later / me.max_hp
                if dealt >= ada.hp:
                    score += 1.0
                if taken >= me.hp:
                    score -= 1.0
                total += score
            values[move] = round(total / SIMS, 3)
        self.sim_values = values
        return values

    # ------------------------------------------------------------ deciding

    def score_moves(self, p, mode, state, rng):
        bias = MODE_BIAS.get(mode, {})
        sims = self.lookahead(p, state, rng) if self.use_lookahead else {}
        u = {}
        sharp = min(0.6, max(0.0, (self.director.intensity - 1.0) * 2))
        for move in MOVES:
            belief = 0.0
            for kind, odds in self.kind_odds.items():
                guess = (1 - sharp) * self.bandit.sample(move, kind, rng) + sharp * self.bandit.mean(move, kind)
                belief += odds * guess
            if self.use_lookahead:
                model = min(1.0, max(0.0, 0.5 + 2 * sims.get(move, -0.25)))
                value = MODEL_WEIGHT * model + (1 - MODEL_WEIGHT) * belief
            else:
                value = belief
            u[move] = value * bias.get(move, 1.0)

        self.harmless = round(self.kind_odds.get("blind", 0.0)
                              + (self.kind_odds.get("repeat", 0.0) if self.is_faded(state) else 0.0), 3)
        if not self.use_lookahead:
            u["strike"] += 0.7 * self.harmless
        if self.last_hit is None or self.is_faded(state):
            u["fade"] = 0.0
        if "shifted" in state.foe.status:
            u["shift"] *= 0.3
        if p.ada_mp < 6:
            u["doubt"] *= 0.3
        elif p.ada_hp_ratio < 0.5 and p.ada_mp <= STARVE_MP:
            u["doubt"] += 0.9
        real = [c for c in state.ada_casts if not c.scry]
        u["echo"] *= min(1.0, (real[-1].damage if real else 0) / 40) or 0.1
        if p.ada_hp_ratio < 0.35:
            u["strike"] += 0.4 * self.director.intensity
            u["echo"] += 0.2 * self.director.intensity
        u["heal"] = 1.6 * (1 - p.my_hp_ratio) ** 2 * bias.get("heal", 1.0) if p.my_hp_ratio < 0.6 else 0.0
        self.utility = {m: round(max(0.0, v), 3) for m, v in u.items()}

    def next_mode(self, p, rng):
        sure = max(self.kind_odds.values(), default=0.0)
        match self.mode:
            case _ if p.my_hp_ratio < 0.3:
                return "unravelling"
            case _ if len(self.history) + len(self.old_history) < LISTEN_UNTIL:
                return "listening"
            case _ if p.ada_hp_ratio < 0.35:
                return "hunting"
            case _ if sure >= CONFIDENT:
                return "mirroring"
            case _:
                return "doubting"

    def idle_movement(self):
        match self.mode:
            case "hunting":
                return Movement("pursue", "ada")
            case "unravelling":
                return Movement("evade", "ada")
            case _:
                return Movement("wander", "home")

    def decide(self, state, rng):
        self.turns += 1
        notes = self.learn(state)
        self.predict(state)
        p = self.percept = self.perception.sense(state)
        old_mode = self.mode
        self.mode = self.next_mode(p, rng)
        self.director.update(p.ada_hp_ratio, p.my_hp_ratio)
        self.score_moves(p, self.mode, state, rng)
        ctx = Context(p, rng)
        intent = self.think.think(ctx, self.mode) or REST
        self.movement = ctx.movement or self.idle_movement()
        self.last_move = intent.action
        self.last_ada_hp = state.ada.hp
        self.free_defence = self.harmless >= 0.8 and intent.action != "fade"
        word = next((w for w in TELL_WORDS if self.top_guess and w in self.top_guess.split("+")), "")
        info = {"mode": self.mode, "move": intent.action, "word": word, "right": self.last_right,
                "hits": self.hits, "guesses": self.guesses, "blind": self.kind_odds.get("blind", 0),
                "my_hp": p.my_hp_ratio, "ada_hp": p.ada_hp_ratio}
        self.says = self.barker.speak(info, self.turns, rng)
        self.last_right = None
        top = sorted(self.prediction.items(), key=lambda kv: -kv[1])[:3]
        rate = f"{self.hits}/{self.guesses}" if self.guesses else "no guesses checked yet"
        self._last = {
            "technique": self.technique,
            "mode": f"{old_mode} -> {self.mode}",
            "sees": summary(p),
            "learned": "; ".join(notes) or "nothing new",
            "predicts": ", ".join(f"{t} {q:.0%}" for t, q in top),
            "kinds": ", ".join(f"{k} {q:.0%}" for k, q in self.kind_odds.items()),
            "uncertainty": f"{entropy(self.prediction):.2f} bits, N-gram order "
                           f"{self.model.order_used(self.old_history + self.history)}",
            "hit rate": rate,
            "harmless": f"P(her next cast can't touch me) = {self.harmless:.0%}",
            "lookahead": ", ".join(f"{m} {v:+.2f}" for m, v in
                                   sorted(self.sim_values.items(), key=lambda kv: -kv[1])),
            "utilities": ", ".join(f"{m} {v:.2f}" for m, v in
                                   sorted(self.utility.items(), key=lambda kv: -kv[1])),
            "beliefs": self.belief_line(),
            "plan": self.think.describe(),
            "steps": " / ".join(ctx.trace),
            "intent": intent.action,
            "movement": f"{self.movement.behaviour} {self.movement.target}",
            "director": self.director.line(),
            "says": self.says or "",
        }
        return intent

    def belief_line(self):
        moved = sorted(((self.bandit.learned(m, k), m, k) for m in MOVES for k in KINDS),
                       key=lambda t: -abs(t[0]))[:3]
        return ", ".join(f"{m} vs {k} {d:+.2f}" for d, m, k in moved if abs(d) > 0.005) \
            or "still my starting guesses"


def find_last_hit(casts):
    for cast in reversed(casts):
        if not cast.scry and cast.damage > 0:
            return signature(cast.runes)
    return None
