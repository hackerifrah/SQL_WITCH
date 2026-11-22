"""Test the Count and NULL with scripted players (design doc section 9).

    python -m sim.evaluate                    # 200 battles per row
    python -m sim.evaluate --n 300 --plots    # charts go in docs/eval/

Every player fights through the real game. Each boss should beat the player who ignores its
lesson and lose to the one who uses it. Ablations switch parts of the AI off to show what they do.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from ai.base import RandomBrain
from ai.count import count_legal_intents
from ai.null import NullBrain, null_legal_intents
from magic import specials
from magic.battle import Battle
from magic.guard import Miscast

MAX_ROUNDS = 25
ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ what a player can read

def catalogue(battle: Battle) -> list[tuple]:
    """Ada's spells, as a player reads them with a free look (element, power, cost, kind)."""
    return battle.conn.execute("SELECT element, power, mp_cost, kind FROM spells").fetchall()


def best(spells, mp: int, kind: str, element: str | None = None, max_cost: int = 999):
    """(element, power) of the strongest affordable spell, or None."""
    options = [(p, -c, e) for e, p, c, k in spells
               if k == kind and c <= min(mp, max_cost) and (element is None or e == element)]
    if not options:
        return None
    p, _, e = max(options)
    return e, p


# ------------------------------------------------------------------ Count Aggregor players

def count_brute(battle: Battle, rng: random.Random) -> str:
    """Ignores the bats: the strongest Light spell at the Count, every turn."""
    pick = best(catalogue(battle), battle.state.ada.mp, "attack", "Light")
    if pick is None:
        return "SELECT name, hp FROM enemies"
    return f"SELECT name FROM spells WHERE element = 'Light' AND kind = 'attack' AND power = {pick[1]}"


def count_joiner(battle: Battle, rng: random.Random) -> str:
    """Has JOIN, never groups: one row per (spell, bat), so the page overflows."""
    return ("SELECT s.name, e.squad FROM spells AS s JOIN enemies AS e ON s.element = e.weakness "
            "WHERE s.kind = 'area'")


def _count_smart(battle: Battle, rng: random.Random, having: bool) -> str:
    state = battle.state
    spells = catalogue(battle)
    ada = state.ada
    if ada.hp_ratio < 0.35:
        heal = best(spells, ada.mp, "heal")
        if heal:
            return f"SELECT name FROM spells WHERE kind = 'heal' AND power = {heal[1]} AND element = '{heal[0]}'"
    if specials.quorum(state):
        gate = " HAVING COUNT(*) >= 3" if having else ""
        return ("SELECT s.name, e.squad FROM spells AS s JOIN enemies AS e ON s.element = e.weakness "
                f"WHERE s.name = 'Holy Flash' GROUP BY e.squad{gate}")
    pick = best(spells, ada.mp, "attack", "Light")
    if pick is None:
        return "SELECT name, hp FROM enemies"
    return f"SELECT name FROM spells WHERE element = 'Light' AND kind = 'attack' AND power = {pick[1]}"


def count_grouper(battle, rng):
    """Copied GROUP BY: one Holy Flash per squad, then strikes the Count in the window."""
    return _count_smart(battle, rng, having=False)


def count_expert(battle, rng):
    """GROUP BY and HAVING: only the squads that hold the quorum, so no MP is wasted."""
    return _count_smart(battle, rng, having=True)


# ------------------------------------------------------------------ NULL players

VOID_JOIN = ("SELECT {distinct}s.name, e.name FROM spells AS s JOIN enemies AS e "
             "ON s.element = {on} WHERE e.hp IS NULL AND s.kind = 'attack' AND s.power = {p}{tail}")
VARIANTS = [   # different technique signatures that all reach NULL
    dict(distinct="", on="COALESCE(e.weakness, 'Arcane')", tail=""),
    dict(distinct="", on="COALESCE(e.weakness, 'Arcane')", tail=" LIMIT 1"),
    dict(distinct="", on="COALESCE(e.weakness, 'Arcane')", tail=" GROUP BY s.name"),
    dict(distinct="DISTINCT ", on="COALESCE(e.weakness, 'Arcane')", tail=""),
    dict(distinct="", on="COALESCE(e.weakness, 'Arcane')", tail=" GROUP BY s.name HAVING COUNT(*) = 1"),
]


def _null_cast(battle: Battle, variant: dict) -> str:
    state, ada = battle.state, battle.state.ada
    spells = catalogue(battle)
    if ada.hp_ratio < 0.3:
        heal = best(spells, ada.mp, "heal")
        if heal:
            return f"SELECT name FROM spells WHERE kind = 'heal' AND power = {heal[1]} AND element = '{heal[0]}'"
    element = state.foe.weakness or "Arcane"          # what a free look at enemies shows
    pick = best(spells, ada.mp, "attack", element)
    if pick is None:
        return "SELECT name, element, weakness FROM enemies"
    return VOID_JOIN.format(p=pick[1], **variant)


def null_blind(battle, rng):
    """Never learns IS NULL: a hard-coded Arcane spell, which passes through the void."""
    pick = best(catalogue(battle), battle.state.ada.mp, "attack", "Arcane")
    if pick is None:
        return "SELECT name FROM enemies"
    return f"SELECT name FROM spells WHERE element = 'Arcane' AND kind = 'attack' AND power = {pick[1]}"


def null_repetitive(battle, rng):
    """Uses COALESCE well, but always the same way."""
    return _null_cast(battle, VARIANTS[0])


def null_alternating(battle, rng):
    """Two techniques in turn: a pattern a 3-gram reads at once."""
    return _null_cast(battle, VARIANTS[len(battle.state.ada_casts) % 2])


def null_varied(battle, rng):
    """Never casts the same way twice in a row, in no fixed order (design doc 5.9)."""
    last = battle.state.ada_casts[-1].runes if battle.state.ada_casts else None
    while True:
        variant = rng.choice(VARIANTS)
        sql = _null_cast(battle, variant)
        if last is None or len(VARIANTS) == 1:
            return sql
        if variant is not VARIANTS[0] or last != ("COALESCE", "IS NULL", "JOIN", "WHERE"):
            return sql


PLAYERS = {
    "count": {"brute": count_brute, "joiner": count_joiner, "grouper": count_grouper,
              "expert": count_expert},
    "null": {"blind": null_blind, "repetitive": null_repetitive,
             "alternating": null_alternating, "varied": null_varied},
}


# ------------------------------------------------------------------ one battle

def play(guardian: str, player, seed: int, brain=None) -> dict:
    battle = Battle(seed=seed, guardian=guardian, brain=brain)
    rng = random.Random(seed * 7919 + 1)                  # the player's own dice
    moves, think_ms, blocked = [], [], Counter()
    while battle.whose_turn != "over" and battle.state.turn <= MAX_ROUNDS:
        if battle.whose_turn == "foe":
            t0 = time.perf_counter()
            turn = battle.foe_act()
            think_ms.append((time.perf_counter() - t0) * 1000)
            moves.append(turn.reasoning.get("intent") or turn.reasoning.get("picked", "?"))
            continue
        sql = player(battle, rng)
        try:
            report = battle.ada_cast(sql)
        except Miscast:
            report = battle.ada_cast("SELECT name, hp FROM enemies")
        for event in report.events:
            if event[0] == "say" and event[2] == "deflect":
                kind = "quorum" if "quorum" in event[1] else ("fade" if "remembers" in event[1]
                                                               else "veil")
                blocked[kind] += 1
        while battle.whose_turn == "ada" and not report.ends_turn:    # a look: now cast
            sql = player(battle, rng)
            try:
                report = battle.ada_cast(sql)
            except Miscast:
                break
    brain = battle.brain
    out = {"won": battle.winner == "ada", "lost": battle.winner == "foe",
           "rounds": battle.state.turn, "ada_hp": battle.state.ada.hp,
           "foe_hp": battle.state.foe.hp, "moves": moves, "think_ms": think_ms,
           "blocked": blocked}
    if isinstance(brain, NullBrain):
        out["predictions"] = list(brain.guess_log)
    return out


def entropy_bits(counts: Counter) -> float:
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values() if c) if n else 0.0


def run(guardian: str, player_name: str, n: int, seed0: int = 1, make_brain=None) -> dict:
    player = PLAYERS[guardian][player_name]
    results = [play(guardian, player, seed0 + i, make_brain() if make_brain else None)
               for i in range(n)]
    moves = Counter(m for r in results for m in r["moves"])
    openings = {tuple(r["moves"][:4]) for r in results}
    blocked = Counter()
    for r in results:
        blocked.update(r["blocked"])
    summary = {
        "guardian": guardian, "player": player_name, "n": n,
        "win": sum(r["won"] for r in results) / n,
        "loss": sum(r["lost"] for r in results) / n,
        "rounds": statistics.mean(r["rounds"] for r in results),
        "ada_hp": statistics.mean(r["ada_hp"] for r in results),
        "entropy": entropy_bits(moves),
        "openings": len(openings),
        "moves": moves,
        "think_ms": statistics.mean(t for r in results for t in r["think_ms"]) if moves else 0,
        "blocked": {k: v / n for k, v in blocked.items()},
    }
    if results and "predictions" in results[0]:
        by_turn = defaultdict(list)
        for r in results:
            for i, pred in enumerate(r.get("predictions", [])):
                by_turn[i + 1].append(pred["correct"])
        summary["accuracy_by_guess"] = {k: sum(v) / len(v) for k, v in sorted(by_turn.items())}
        running = {}
        for k in sorted(by_turn):
            seen = [x for j in by_turn if j <= k for x in by_turn[j]]
            running[k] = sum(seen) / len(seen)
        summary["running_accuracy"] = running
        allp = [p["correct"] for r in results for p in r.get("predictions", [])]
        summary["accuracy"] = sum(allp) / len(allp) if allp else 0.0
    return summary


# ------------------------------------------------------------------ the report

def table(rows: list[dict], extra: list[str] = ()) -> str:
    head = ["Guardian", "Player", "Ada wins", "Ada loses", "Rounds", "Ada HP left",
            "Move entropy (bits)", "Distinct openings", "ms/turn", *extra]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["guardian"], r["player"], f"{r['win']:.0%}", f"{r['loss']:.0%}",
                 f"{r['rounds']:.1f}", f"{r['ada_hp']:.0f}", f"{r['entropy']:.2f}",
                 str(r["openings"]), f"{r['think_ms']:.2f}"]
        for key in extra:
            value = r.get(key)
            cells.append(f"{value:.0%}" if isinstance(value, float) else str(value or ""))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Count Aggregor and NULL headless.")
    parser.add_argument("--n", type=int, default=200, help="battles per experiment")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plots", action="store_true", help="write PNG plots to docs/eval/")
    parser.add_argument("--out", default=None, help="also write the markdown report here")
    args = parser.parse_args(argv)
    n, s = args.n, args.seed
    report = [f"# Evaluation: Count Aggregor and NULL ({n} seeded battles per row)\n"]

    count_rows = [run("count", p, n, s) for p in PLAYERS["count"]]
    report += ["## Count Aggregor (behaviour tree + fuzzy logic)\n", table(count_rows), ""]
    moves = count_rows[-1]["moves"]
    report.append("Moves against the expert: " + ", ".join(
        f"{m} {c / sum(moves.values()):.0%}" for m, c in moves.most_common()) + "\n")

    null_rows = [run("null", p, n, s) for p in PLAYERS["null"]]
    report += ["## NULL (N-gram prediction + lookahead + Thompson sampling)\n",
               table(null_rows, ["accuracy"]), "",
               "`accuracy`: how often NULL's top prediction of Ada's next technique was right.\n"]

    ablations = [
        {**run("count", "expert", n, s, lambda: RandomBrain(count_legal_intents)),
         "guardian": "count (random baseline)"},
        {**run("count", "grouper", n, s, lambda: RandomBrain(count_legal_intents)),
         "guardian": "count (random baseline)"},
        {**run("null", "repetitive", n, s, lambda: RandomBrain(null_legal_intents)),
         "guardian": "null (random baseline)"},
        {**run("null", "repetitive", n, s, lambda: NullBrain(learning=False)),
         "guardian": "null (no learning)"},
        {**run("null", "repetitive", n, s, lambda: NullBrain(n=1)),
         "guardian": "null (1-gram only)"},
        {**run("null", "varied", n, s), "guardian": "null"},
        {**run("null", "varied", n, s, lambda: NullBrain(lookahead=False)),
         "guardian": "null (no lookahead)"},
    ]
    report += ["## Ablations\n",
               "The same players against weaker versions of each brain. A real brain should beat "
               "the random baseline, and NULL should be stronger with learning and with a longer "
               "context than without.\n",
               table([count_rows[-1], ablations[0], count_rows[-2], ablations[1], null_rows[1],
                      *ablations[2:]],
                     ["accuracy"]), ""]

    for row in null_rows:
        if row["player"] in ("repetitive", "alternating", "varied"):
            curve = row.get("accuracy_by_guess", {})
            points = ", ".join(f"{k}: {v:.0%}" for k, v in list(curve.items())[:10])
            report.append(f"- NULL learning curve vs {row['player']}: {points}")
    text = "\n".join(report)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if args.plots:
        plot(count_rows, null_rows, ablations)
    return 0


def plot(count_rows, null_rows, ablations) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ROOT / "docs" / "eval"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for row in null_rows:
        if row["player"] == "blind":                # its fights are too short to draw a curve
            continue
        curve = row.get("running_accuracy", {})
        xs = [k for k in curve if k <= 12]
        if xs:
            ax.plot(xs, [curve[k] for k in xs], marker="o", label=f"{row['player']} player")
    ax.axhline(0.25, color="grey", linestyle="--", linewidth=1, label="random guess")
    ax.set_xlabel("predictions made so far in the battle")
    ax.set_ylabel("share of correct predictions")
    ax.set_ylim(0, 1)
    ax.set_title("NULL reads patterns; varied play stays unpredictable")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "null_learning_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    rows = count_rows + null_rows
    labels = [f"{r['guardian']}\n{r['player']}" for r in rows]
    ax.bar(range(len(rows)), [r["win"] for r in rows], color="#5b6ee1")
    ax.set_xticks(range(len(rows)), labels, fontsize=7)
    ax.set_ylabel("Ada's win rate")
    ax.set_ylim(0, 1)
    ax.set_title("Each guardian beats players who ignore its lesson")
    fig.tight_layout()
    fig.savefig(out / "win_rates.png", dpi=150)
    plt.close(fig)
    print(f"\nplots written to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
