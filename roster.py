"""The guardians of this build, in the order Ada meets them (design doc 5.2).

Each Guardian bundles what a battle needs: which foe row, Ada's level for the fight, the
brain, the SQL style, the runes the guardian can write and the one it teaches.
Never imports pygame.

STATUS: the Asterisk Slime, Sir Wherewell and the tutorial's Training Dummy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ai import count as count_ai
from ai import null as null_ai
from ai.dummy import DummyBrain, render_dummy
from ai.fsm import SlimeBrain
from ai.knight import ACTION_SPELLS, KnightBrain
from ai.styles import (COUNT_ALSO_TEACHES, COUNT_RUNES, COUNT_SIGNATURE, KNIGHT_RUNES,
                       KNIGHT_SIGNATURE, NULL_ALSO_TEACHES, NULL_RUNES, NULL_SIGNATURE,
                       SLIME_RUNES, SLIME_SIGNATURE, render_count, render_knight, render_null,
                       render_slime)
from magic import specials


@dataclass(frozen=True)
class Guardian:
    key: str
    foe_id: int
    level: int                                   # Ada's level in this fight (data/levels.csv)
    runes: frozenset[str]                        # what the guardian can write
    signature: str                               # the rune the Invitation hands to Ada
    ada_runes: frozenset[str]                    # what Ada already knows when it starts
    style: Callable
    make_brain: Callable                         # (sqlite connection) -> brain
    setup: Callable | None = None
    also_teaches: frozenset[str] = frozenset()


def _slime_brain(conn) -> SlimeBrain:
    cheapest = conn.execute("SELECT MIN(mp_cost) FROM foe_spells WHERE foe_id = 1").fetchone()[0]
    return SlimeBrain({"splat": cheapest, "careful": cheapest})


def _knight_brain(conn) -> KnightBrain:
    costs = {}
    for action, spell in ACTION_SPELLS.items():
        row = conn.execute("SELECT mp_cost FROM foe_spells WHERE foe_id = 2 AND name = ?",
                           (spell,)).fetchone()
        if row:
            costs[action] = row[0]
    return KnightBrain(costs)


def _costs(conn, foe_id: int, spells: dict[str, str], defaults: dict[str, int]) -> dict[str, int]:
    costs = dict(defaults)
    for action, spell in spells.items():
        row = conn.execute("SELECT mp_cost FROM foe_spells WHERE foe_id = ? AND name = ?",
                           (foe_id, spell)).fetchone()
        if row:
            costs[action] = row[0]
    return costs


def _count_brain(conn) -> count_ai.CountBrain:
    return count_ai.CountBrain(_costs(conn, count_ai.COUNT_ID, count_ai.ACTION_SPELLS,
                                      count_ai.DEFAULT_COSTS))


def _null_brain(conn) -> null_ai.NullBrain:
    return null_ai.NullBrain(_costs(conn, null_ai.NULL_ID, null_ai.ACTION_SPELLS,
                                    null_ai.DEFAULT_COSTS))


COUNT_ADA_RUNES = frozenset({"LIMIT", "WHERE", "JOIN"})
NULL_ADA_RUNES = COUNT_ADA_RUNES | {"GROUP BY", "HAVING", "DISTINCT"}

GUARDIANS = {
    "slime": Guardian("slime", 1, 1, frozenset(SLIME_RUNES), SLIME_SIGNATURE, frozenset(),
                      render_slime, _slime_brain),
    "knight": Guardian("knight", 2, 2, frozenset(KNIGHT_RUNES), KNIGHT_SIGNATURE,
                       frozenset({"LIMIT"}), render_knight, _knight_brain),
    # The tutorial's partner. It teaches no rune (an empty signature never matches a query's
    # runes), so Ada leaves the tutorial knowing exactly what she knew before.
    "dummy": Guardian("dummy", 3, 1, frozenset(), "", frozenset(), render_dummy,
                      lambda conn: DummyBrain()),
    "count": Guardian("count", count_ai.COUNT_ID, 4, frozenset(COUNT_RUNES), COUNT_SIGNATURE,
                      COUNT_ADA_RUNES, render_count, _count_brain, specials.setup_count,
                      COUNT_ALSO_TEACHES),
    "null": Guardian("null", null_ai.NULL_ID, 5, frozenset(NULL_RUNES), NULL_SIGNATURE,
                     NULL_ADA_RUNES, render_null, _null_brain, specials.setup_null,
                     NULL_ALSO_TEACHES),
}
# twins go after the knight: ["slime", "knight", "twins", "count", "null"]
ORDER = ["slime", "knight", "count", "null"]


def next_guardian(key: str) -> str | None:
    i = ORDER.index(key)
    return ORDER[i + 1] if i + 1 < len(ORDER) else None
