"""Count Aggregor: a behaviour tree guardian with fuzzy logic and a flock of Tally Bats.

Turn: perceive (HP, MP, threat + his swarm) -> update his ledger of how Ada casts ->
stance (match/case) -> fuzzy aggression, scaled by the AI Director -> tick the behaviour tree
-> the chosen leaf runs a goal plan -> Intent + movement -> a bark about what he noticed.
While a squad has 3+ bats (the quorum) he can't be hurt.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from ai import btree
from ai.barks import Bark, Barker
from ai.base import Intent
from ai.btree import Blackboard, Status
from ai.director import Director
from ai.fuzzy import FuzzyRules, FuzzyVar, left_shoulder, right_shoulder, triangle
from ai.goals import AtomicGoal, CompositeGoal, Context
from ai.guardian import GuardianBrain
from ai.perception import summary
from ai.steering import Movement
from magic.specials import MAX_BATS, QUORUM

COUNT_ID = 10

ACTION_SPELLS = {
    "bolt": "Tithe Bolt", "summon": "Call the Tally", "muster": "Muster the Ranks",
    "rally": "Roll Call", "swarm": "Swarm of Sums", "feed": "Blood Tally",
    "ward": "Abacus Shield",
}
DEFAULT_COSTS = {"bolt": 4, "summon": 6, "muster": 6, "rally": 3, "swarm": 5, "feed": 5,
                 "ward": 4}
COUNT_INTENTS = {
    "bolt": Intent("bolt", "Ada", "attack"),
    "summon": Intent("summon", None, "summon"),
    "muster": Intent("muster", None, "summon"),
    "rally": Intent("rally", None, "rally"),
    "swarm": Intent("swarm", "Ada", "attack"),
    "feed": Intent("feed", None, "heal"),
    "ward": Intent("ward", None, "ward"),
    "roll_call": Intent("roll_call", None, "scry"),
    "rest": Intent("rest", None, "scry"),
}

LOW_HP = 0.35
ADA_LOW = 0.30
SUMMON_COOLDOWN = 1
AGGREGATING = 0.5


def count_legal_intents(state):
    return list(COUNT_INTENTS.values())


# ------------------------------------------------------------------ what he sees

@dataclass(frozen=True)
class SwarmInfo:
    bats: int
    squads: dict
    quorum: tuple
    stragglers: int
    weak_squads: int

    @property
    def can_regroup(self):
        return self.weak_squads >= 2 and self.stragglers >= QUORUM

    def line(self):
        groups = ", ".join(f"{k} {v}" for k, v in sorted(self.squads.items())) or "none"
        return f"{self.bats} bats ({groups}) quorum: {', '.join(self.quorum) or 'BROKEN'}"


def look_at_swarm(state):
    groups = Counter(m.squad for m in state.minions if m.squad and m.alive)
    weak = {k: v for k, v in groups.items() if v < QUORUM}
    return SwarmInfo(bats=sum(groups.values()), squads=dict(groups),
                     quorum=tuple(sorted(k for k, v in groups.items() if v >= QUORUM)),
                     stragglers=sum(weak.values()), weak_squads=len(weak))


@dataclass
class Ledger:
    """The Count counts Ada too: which runes she uses, how often she groups, overflows, kills."""
    seen: int = 0
    casts: int = 0
    runes: Counter = field(default_factory=Counter)
    grouped: int = 0
    overflowed: int = 0
    bats_before: int | None = None
    kills: list = field(default_factory=list)

    def update(self, state, bats_now):
        for cast in state.ada_casts[self.seen:]:
            if cast.scry:
                continue
            self.casts += 1
            self.runes.update(cast.runes)
            if {"GROUP BY", "DISTINCT"} & set(cast.runes):
                self.grouped += 1
            if cast.overflow:
                self.overflowed += 1
        if self.bats_before is not None and len(state.ada_casts) > self.seen:
            self.kills.append(max(0, self.bats_before - bats_now))
        self.seen = len(state.ada_casts)

    @property
    def group_rate(self):
        return self.grouped / self.casts if self.casts else 0.0

    @property
    def kill_rate(self):
        last = self.kills[-3:]
        return sum(last) / len(last) if last else 0.0

    def line(self):
        top = ", ".join(f"{r} x{n}" for r, n in self.runes.most_common(3)) or "nothing yet"
        return (f"{self.casts} casts: {top} | groups {self.group_rate:.0%} | "
                f"overflowed {self.overflowed} | kills/cast {self.kill_rate:.1f}")


# ------------------------------------------------------------------ fuzzy aggression

SWARM_SIZE = FuzzyVar("swarm", {
    "few": lambda x: left_shoulder(x, 2, 5),
    "some": lambda x: triangle(x, 3, 6, 9),
    "lots": lambda x: right_shoulder(x, 7, 10),
})
ADA_HEALTH = FuzzyVar("ada", {
    "low": lambda x: left_shoulder(x, 0.25, 0.5),
    "ok": lambda x: triangle(x, 0.3, 0.55, 0.8),
    "high": lambda x: right_shoulder(x, 0.6, 0.9),
})
MY_HEALTH = FuzzyVar("me", {
    "low": lambda x: left_shoulder(x, 0.25, 0.5),
    "ok": lambda x: triangle(x, 0.3, 0.55, 0.8),
    "high": lambda x: right_shoulder(x, 0.6, 0.9),
})
AGGRESSION_RULES = [
    ({"swarm": "lots", "ada": "low"}, "reckless"),
    ({"swarm": "lots", "me": "high"}, "reckless"),
    ({"ada": "low"}, "reckless"),
    ({"swarm": "some", "ada": "ok"}, "steady"),
    ({"swarm": "some", "me": "ok"}, "steady"),
    ({"swarm": "few", "me": "high"}, "steady"),
    ({"swarm": "few", "me": "low"}, "careful"),
    ({"me": "low", "ada": "high"}, "careful"),
    ({"swarm": "few", "ada": "high"}, "careful"),
]
AGGRESSION = {"careful": 0.15, "steady": 0.5, "reckless": 0.85}

STANCE_BIAS = {   # stance -> multipliers for (bite, tithe, drink)
    "counting": (1.0, 1.0, 1.0),
    "recounting": (0.5, 1.3, 1.0),
    "frenzied": (1.4, 0.6, 0.3),
    "audited": (0.8, 1.2, 1.0),
    "gorging": (0.6, 0.6, 1.8),
}

COUNT_BARKS = [
    Bark("split", lambda i: i["mode"] == "audited" and i["plan"] in ("summon", "expand"),
         ("You GROUP BY? Then I split my ranks.", "Group this, witch. More squads!"), 6),
    Bark("overflow", lambda i: i["overflowed"],
         ("Your page can't hold my army, witch.", "Overflowing? My bats love spilled mana."), 5),
    Bark("big kill", lambda i: i["kills"] >= 4,
         ("{kills} bats in one spell? ...Noted.", "You counted well. So will I."), 5),
    Bark("broken", lambda i: not i["quorum"],
         ("Stragglers, fall in! The quorum!", "You broke my quorum? Recount!"), 4),
    Bark("hurt", lambda i: i["my_hp"] < 0.35,
         ("The ledger is bleeding red...", "A sip from the swarm, for my books."), 3),
    Bark("finish", lambda i: i["ada_hp"] < 0.3,
         ("{bats} bites. Count them as you fall.", "Your tally ends here, witch."), 3),
    Bark("reckless", lambda i: i["aggression"] > 0.7,
         ("Bite, my sums! All {bats} of you!", "Swarm of sums: add them up!"), 2),
    Bark("idle", lambda i: True,
         ("I count {bats} bats. Zero chances.", "{squads} squads, {bats} bats. Mine.",
          "Tithe time, witch.", "One, two, three... I never lose count."), 1),
]


# ------------------------------------------------------------------ plans

MOVES = {
    "bolt": ("stride out", Movement("arrive", "front")),
    "swarm": ("loose the swarm", Movement("seek", "front")),
    "summon": ("raise his arms", Movement("arrive", "home")),
    "muster": ("raise his arms", Movement("arrive", "home")),
    "rally": ("call the roll", Movement("arrive", "home")),
    "feed": ("draw them close", Movement("arrive", "home")),
    "ward": ("step back", Movement("arrive", "home")),
    "roll_call": ("pace the ranks", Movement("wander", "home")),
    "rest": ("pace the ranks", Movement("wander", "home")),
}


def can_do(action, p, swarm):
    """MP, and the swarm has to match the HAVING his query prints (so the cast isn't empty)."""
    if action in ("roll_call", "rest"):
        return True
    if not p.can(action):
        return False
    match action:
        case "swarm":
            return swarm.bats >= QUORUM
        case "feed":
            return swarm.bats > 0 and p.my_hp_ratio < 1.0
        case "summon" | "muster":
            return swarm.bats < MAX_BATS
        case "rally":
            return swarm.can_regroup
        case _:
            return True


def make_plan(name, actions, brain):
    move_name, movement = MOVES[actions[0]]
    steps = [AtomicGoal(move_name, movement=movement)]
    for a in actions:
        steps.append(AtomicGoal(a, COUNT_INTENTS[a], MOVES[a][1],
                                requires=lambda p, a=a: can_do(a, p, brain.swarm)))
    return CompositeGoal(name, steps)


class PlanLeaf(btree.Node):
    """Leaf that runs a goal plan. RUNNING while the plan has more turns to go."""

    def __init__(self, name, actions, brain):
        super().__init__(name)
        self.actions = actions
        self.brain = brain

    def run(self, bb):
        ctx = bb["ctx"]
        plans = bb["plans"]
        plan = plans.get(self.name)
        if plan is None or plan.status.name in ("COMPLETED", "FAILED"):
            actions = self.actions(bb) if callable(self.actions) else self.actions
            plan = plans[self.name] = make_plan(self.name, actions, self.brain)
            bb.log(f"  new plan {self.name}: {' > '.join(actions)}")
        else:
            bb.log(f"  resume plan {self.name}")
        plan.step(ctx)
        if ctx.intent is None:
            plans.pop(self.name, None)
            return Status.FAILURE
        bb["acted"] = self.name
        if plan.status.name == "ACTIVE":
            return Status.RUNNING
        plans.pop(self.name, None)
        return Status.SUCCESS

    def label(self):
        return f"[{self.name}]"


# ------------------------------------------------------------------ the tree

def build_tree(brain):
    B = btree

    def p(bb):
        return bb["percept"]

    def sw(bb):
        return bb["swarm"]

    def aggr(bb):
        return bb["aggression"]

    def summon_kind(bb):
        if bb["mode"] == "audited" or sw(bb).bats == 0:
            return ["summon"]
        return ["muster" if bb.rng.random() < 0.5 else "summon"]

    def tally_target(bb):
        return min(MAX_BATS, 8 + round(2 * bb["ledger"].kill_rate / QUORUM))

    def bite_weight(bb):
        return (0.2 + aggr(bb)) * min(1.0, sw(bb).bats / 6) * STANCE_BIAS[bb["mode"]][0]

    def tithe_weight(bb):
        return (1.1 - aggr(bb)) * 0.6 * STANCE_BIAS[bb["mode"]][1]

    def drink_weight(bb):
        return (1 - aggr(bb)) * 2 * (1 - p(bb).my_hp_ratio) * STANCE_BIAS[bb["mode"]][2]

    def leaf(name, actions):
        return PlanLeaf(name, actions, brain)

    summon = B.Cooldown("summon cooldown", SUMMON_COOLDOWN, leaf("summon", summon_kind), key="summon")
    return B.Selector(
        "Count's will",
        B.Sequence("opening",
                   B.Condition("first turn", lambda bb: bb.turn == 1),
                   leaf("roll call", ["roll_call"])),
        B.Sequence("survive",
                   B.Condition("badly hurt", lambda bb: p(bb).my_hp_ratio < LOW_HP),
                   B.Condition("has a swarm", lambda bb: sw(bb).bats >= 2),
                   B.Chance("will to live", 0.7,
                            B.RandomSelector("mend", [0.65, 0.35],
                                             leaf("feed", ["feed"]), leaf("fortify", ["ward"])))),
        B.Sequence("reinforce",
                   B.Condition("quorum broken", lambda bb: not sw(bb).quorum),
                   B.Selector("restore quorum",
                              B.Sequence("stragglers",
                                         B.Condition("stragglers can regroup",
                                                     lambda bb: sw(bb).can_regroup),
                                         leaf("regroup", ["rally", "summon"])),
                              summon)),
        B.Sequence("punish",
                   B.Condition("Ada is open", lambda bb: p(bb).ada_exhausted or p(bb).ada_overflowing),
                   B.Condition("swarm can bite", lambda bb: sw(bb).bats >= QUORUM),
                   leaf("pounce", ["swarm"])),
        B.Sequence("execute",
                   B.Condition("Ada is weak", lambda bb: p(bb).ada_hp_ratio < ADA_LOW),
                   B.UtilitySelector("finisher",
                                     [lambda bb: 4 * sw(bb).bats / max(1, p(bb).ada_hp),
                                      lambda bb: 26 / max(1, p(bb).ada_hp)],
                                     leaf("swarm finish", ["swarm"]), leaf("bolt finish", ["bolt"]))),
        B.Sequence("expand",
                   B.Condition("below his tally target", lambda bb: sw(bb).bats < tally_target(bb)),
                   B.Chance("ambition",
                            lambda bb: 0.2 + 0.35 * (1 - aggr(bb)) + 0.1 * bb["ledger"].kill_rate / QUORUM,
                            B.Cooldown("summon cooldown", SUMMON_COOLDOWN, leaf("expand", summon_kind),
                                       key="summon"))),
        B.Sequence("guard",
                   B.Condition("feels threatened",
                               lambda bb: p(bb).threat > 0.12 and p(bb).my_ward < 5),
                   B.Chance("caution", lambda bb: 0.7 * (1 - aggr(bb)), leaf("guard", ["ward"]))),
        B.RandomSelector("harass", [bite_weight, tithe_weight, drink_weight],
                         leaf("bite", ["swarm"]), leaf("tithe", ["bolt"]), leaf("drink", ["feed"])),
        leaf("rest", ["rest"]),
    )


# ------------------------------------------------------------------ the brain

class CountBrain(GuardianBrain):
    technique = "behaviour tree + fuzzy logic + ledger of Ada's casts"
    start_mode = "counting"

    def __init__(self, costs=None):
        super().__init__(costs or dict(DEFAULT_COSTS), evaluators=[])
        self.tree = build_tree(self)
        self.fuzzy = FuzzyRules([SWARM_SIZE, ADA_HEALTH, MY_HEALTH], AGGRESSION_RULES, AGGRESSION)
        self.bb = None
        self.ledger = Ledger()
        self.director = Director()
        self.barker = Barker(COUNT_BARKS)
        self.swarm = look_at_swarm_empty()
        self.turns = 0
        self.swarm_order = "hover"
        self.cooldowns = {}
        self.plans = {}
        self.says = None

    def next_mode(self, p, rng):
        s = self.swarm
        match self.mode:
            case _ if p.my_hp_ratio < LOW_HP:
                return "gorging"
            case _ if not s.quorum and s.bats < QUORUM:
                return "recounting"
            case _ if p.ada_hp_ratio < 0.4 and s.bats >= 6:
                return "frenzied"
            case _ if self.ledger.casts >= 2 and self.ledger.group_rate >= AGGREGATING:
                return "audited"
            case "recounting" | "frenzied" | "audited" | "gorging":
                return "counting"
            case mode:
                return mode

    def idle_movement(self):
        match self.mode:
            case "frenzied":
                return Movement("arrive", "front")
            case "gorging" | "recounting":
                return Movement("evade", "ada")
            case _:
                return Movement("wander", "home")

    def decide(self, state, rng):
        self.turns += 1
        p = self.percept = self.perception.sense(state)
        self.swarm = look_at_swarm(state)
        self.ledger.update(state, self.swarm.bats)
        self.ledger.bats_before = self.swarm.bats
        old_mode = self.mode
        self.mode = self.next_mode(p, rng)
        fuzzy_aggr = self.fuzzy.run(swarm=self.swarm.bats, ada=p.ada_hp_ratio, me=p.my_hp_ratio)
        intensity = self.director.update(p.ada_hp_ratio, p.my_hp_ratio)
        aggression = min(1.0, max(0.0, fuzzy_aggr * intensity))

        ctx = Context(p, rng)
        bb = self.bb = Blackboard(rng, turn=self.turns, cooldowns=self.cooldowns)
        bb.data.update(percept=p, swarm=self.swarm, ledger=self.ledger, mode=self.mode, ctx=ctx,
                       plans=self.plans, aggression=aggression)
        self.tree.tick(bb)
        acted = bb.get("acted")
        for name in list(self.plans):
            if name != acted:
                self.plans.pop(name)
        intent = ctx.intent or COUNT_INTENTS["rest"]
        self.movement = ctx.movement or self.idle_movement()
        self.swarm_order = "bite" if intent.action == "swarm" else "hover"
        kills = self.ledger.kills[-1] if self.ledger.kills else 0
        info = {"mode": self.mode, "plan": acted or "rest", "overflowed": p.ada_overflowing,
                "kills": kills, "quorum": bool(self.swarm.quorum), "my_hp": p.my_hp_ratio,
                "ada_hp": p.ada_hp_ratio, "aggression": aggression, "bats": self.swarm.bats,
                "squads": len(self.swarm.squads)}
        self.says = self.barker.speak(info, self.turns, rng)
        fz = self.fuzzy.last
        self._last = {
            "technique": self.technique,
            "mode": f"{old_mode} -> {self.mode}",
            "sees": summary(p),
            "swarm": self.swarm.line(),
            "ledger": self.ledger.line(),
            "fuzzy": ", ".join(f"{k} {v:.2f}" for k, v in fz["output"].items())
                     + f" -> aggression {fz['value']:.2f}",
            "director": self.director.line(),
            "tried": ", ".join(btree.tried_branches(bb.trace)),
            "tree": " > ".join(btree.success_path(bb.trace)),
            "weights": bb.get("weights", {}),
            "plan": acted or "rest",
            "steps": " / ".join(ctx.trace),
            "intent": intent.action,
            "movement": f"{self.movement.behaviour} {self.movement.target}",
            "says": self.says or "",
        }
        return intent

    def explain(self):
        return dict(self._last)

    def full_trace(self):
        return list(self.bb.trace) if self.bb else []


def look_at_swarm_empty():
    return SwarmInfo(0, {}, (), 0, 0)
