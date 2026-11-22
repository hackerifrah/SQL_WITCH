"""Extra rules for Count Aggregor (bats, squads, the quorum) and NULL (hidden values, the Veil
of Doubt, fade, shift, doubt, echo). For every other guardian these hooks do nothing.

The brains never call this file: every effect comes from a spell row a real query returned.
"""

from __future__ import annotations

from collections import Counter

from magic import rules

QUORUM = 3
MAX_BATS = 12
BAT_HP = 26
BAT_ELEMENT, BAT_WEAKNESS = "Shadow", "Light"
BAT_ID_START = 100
SQUAD_NAMES = ("Left Wing", "Right Wing", "Rearguard", "Vanguard")
START_SQUADS = 2

VEIL = frozenset({"IS NULL", "COALESCE"})
ECHO_CAP = 45

KINDS = frozenset({"summon", "muster", "rally", "swarm", "feed", "doubt", "echo", "shift", "fade"})
SELF_KINDS = frozenset({"summon", "muster", "rally", "feed", "shift", "fade"})


# ------------------------------------------------------------------ bats

def living_bats(state):
    return [m for m in state.minions if m.squad and m.alive]


def squads(state):
    groups = {}
    for bat in living_bats(state):
        groups.setdefault(bat.squad, []).append(bat)
    return groups


def quorum(state):
    return sorted(name for name, bats in squads(state).items() if len(bats) >= QUORUM)


def new_bat(state, squad):
    number = state.foe.status.get("bats_made", 0) + 1
    state.foe.status["bats_made"] = number
    bat = rules.Combatant(f"Tally Bat {number}", BAT_HP, BAT_HP, 0, 0, 0, 7, BAT_ELEMENT,
                          BAT_WEAKNESS, squad=squad)
    bat.status["id"] = BAT_ID_START + number
    state.minions.append(bat)
    return bat


def summon_squad(state, size, split=None):
    """New squad if a name is free (or split=True), otherwise top up the smallest squad."""
    room = MAX_BATS - len(living_bats(state))
    size = max(0, min(size, room))
    groups = squads(state)
    free = [n for n in SQUAD_NAMES if n not in groups]
    if free and (split is None or split or not groups):
        squad = free[0]
    elif groups:
        squad = min(groups, key=lambda n: (len(groups[n]), n))
    else:
        squad = SQUAD_NAMES[0]
    for _ in range(size):
        new_bat(state, squad)
    return squad, size


def rally(state):
    """Squads under the quorum merge into the biggest of them."""
    groups = squads(state)
    weak = [n for n, bats in groups.items() if len(bats) < QUORUM]
    if len(weak) < 2:
        return None, 0
    keep = max(weak, key=lambda n: (len(groups[n]), n))
    moved = 0
    for name in weak:
        if name != keep:
            for bat in groups[name]:
                bat.squad = keep
                moved += 1
    return keep, moved


def setup_count(conn, state, rng):
    for _ in range(START_SQUADS):
        summon_squad(state, rng.randint(3, 5), split=True)
    sync(conn, state)


def setup_null(conn, state, rng):
    state.foe.status.update(veil=True, hidden_hp=True)
    sync(conn, state)


# ------------------------------------------------------------------ hooks used by cast.py

def extra_names(state):
    names = {}
    for bat in living_bats(state):
        names[bat.name] = f"minion:{bat.name}"
        names[bat.squad] = f"squad:{bat.squad}"
    return names


def is_swarm_target(key):
    return key.startswith(("squad:", "minion:"))


def swarm_targets(state, key, kind):
    what, _, name = key.partition(":")
    if what == "minion":
        return [b for b in living_bats(state) if b.name == name]
    bats = [b for b in living_bats(state) if b.squad == name]
    return bats if kind == "area" else bats[:1]


def begin_cast(state, caster):
    foe = state.foe
    if caster == "foe" and "shifted" in foe.status:
        foe.element, foe.weakness = foe.status.pop("shifted")


def blocked(state, target, report):
    if target is state.foe and report.caster == "ada":
        if state.minions and quorum(state):
            return "quorum"
        if target.status.get("veil"):
            if "enemies" not in report.sql.lower() or not VEIL & report.runes:
                return "veil"
            fade = target.status.get("fade")
            if fade is not None and tuple(sorted(report.runes)) == tuple(fade):
                return "fade"
    return None


BLOCK_LINES = {
    "quorum": "{spell} bounces off: the quorum holds!",
    "veil": "{spell} passes through: name the void!",
    "fade": "{spell} fades: it knows that technique!",
}


def block_event(spell, target, reason):
    return ("say", BLOCK_LINES[reason].format(spell=spell, target=target.name), "deflect")


def hit_swarm(row, spell, me, bats, label, chart, rng, report, make_cast):
    total, crit, mult = 0, False, 1.0
    for bat in bats:
        hit = rules.damage(spell.power, spell.element, bat, chart, rng)
        rules.apply_hit(bat, hit)
        total += hit.amount
        crit = crit or hit.crit
        mult = hit.multiplier
        report.casts.append(make_cast(row, spell.name, bat.name, spell.kind, damage=hit.amount,
                                      crit=hit.crit, effective=hit.multiplier, element=spell.element))
    report.events.append(("cast", row, me.name, spell.name, label, total, crit, mult))


def end_cast(state, report):
    if not state.minions:
        return
    fallen = Counter(b.squad for b in state.minions if b.squad and not b.alive)
    state.minions = [m for m in state.minions if m.alive]
    for squad, n in sorted(fallen.items()):
        left = len(squads(state).get(squad, []))
        tail = "scattered" if left == 0 else f"{left} left"
        report.events.append(("say", f"{n} bat{'s' if n > 1 else ''} of {squad} fall ({tail}).", "faint"))
    if fallen and not quorum(state):
        report.events.append(("say", "Quorum broken: the Count can be hurt!", "shatter"))


def apply(spell, row, me, target, state, chart, rng, report, make_cast):
    match spell.kind:
        case "summon" | "muster":
            had_squads = len(squads(state))
            squad, n = summon_squad(state, rng.randint(QUORUM, max(QUORUM, spell.power)),
                                    split=spell.kind == "summon")
            text = f"{spell.name}: {n} bats join {squad}." if n else f"{spell.name}: the barracks are full."
            report.casts.append(make_cast(row, spell.name, me.name, "summon", amount=n))
            report.events.append(("say", text, "cast"))
            if had_squads and n and len(quorum(state)) == 1 and len(squads(state)[squad]) >= QUORUM:
                report.events.append(("say", "The quorum holds again.", "ward_up"))
        case "rally":
            squad, moved = rally(state)
            text = f"{spell.name}: {moved} bats regroup in {squad}." if moved else f"{spell.name}: nobody answers."
            report.casts.append(make_cast(row, spell.name, me.name, "rally", amount=moved))
            report.events.append(("say", text, "cast"))
        case "swarm":
            bats = living_bats(state)
            if not bats:
                report.events.append(("say", f"{spell.name}: no bats to bite.", "fizzle"))
                return
            hit = rules.damage(spell.power * len(bats), spell.element, target, chart, rng)
            rules.apply_hit(target, hit)
            report.casts.append(make_cast(row, spell.name, target.name, "attack", damage=hit.amount,
                                          crit=hit.crit, effective=hit.multiplier, element=spell.element))
            report.events.append(("cast", row, me.name, f"{spell.name} x{len(bats)}", target.name,
                                  hit.amount, hit.crit, hit.multiplier))
            if not target.alive:
                report.events.append(("faint", target.name))
        case "feed":
            amount = rules.heal(me, spell.power * len(living_bats(state)))
            report.casts.append(make_cast(row, spell.name, me.name, "heal", amount=amount))
            report.events.append(("heal", row, spell.name, me.name, amount))
        case "doubt":
            before = target.mp
            target.mp = max(0, target.mp - spell.power)
            report.casts.append(make_cast(row, spell.name, target.name, "doubt", amount=before - target.mp))
            report.events.append(("say", f"{spell.name}: Ada doubts, -{before - target.mp} MP.", "fizzle"))
            report.events.append(("mp", target.name, before, target.mp))
        case "echo":
            power = min(ECHO_CAP, round(spell.power / 100 * last_damage(state)))
            if power <= 0:
                report.events.append(("say", f"{spell.name}: nothing to echo.", "fizzle"))
                return
            hit = rules.damage(power, spell.element, target, chart, rng)
            rules.apply_hit(target, hit)
            report.casts.append(make_cast(row, spell.name, target.name, "attack", damage=hit.amount,
                                          crit=hit.crit, effective=hit.multiplier, element=spell.element))
            report.events.append(("cast", row, me.name, spell.name, target.name, hit.amount, hit.crit,
                                  hit.multiplier))
            if not target.alive:
                report.events.append(("faint", target.name))
        case "shift":
            new, weak = pick_shift(state, chart, rng)
            me.status["shifted"] = (me.element, me.weakness)
            me.element, me.weakness = new, weak
            report.casts.append(make_cast(row, spell.name, me.name, "shift", element=new))
            report.events.append(("say", f"{me.name} turns {new}, weak to {weak}!", "ward_up"))
        case "fade":
            sig = last_hit_runes(state)
            me.status["fade"] = sig
            report.casts.append(make_cast(row, spell.name, me.name, "fade"))
            report.events.append(("say", f"{me.name} memorised your last technique!", "ward_up"))
            if sig:
                report.events.append(("say", "Blocked: " + "+".join(sig)[:51], "ward_up"))


def last_damage(state):
    casts = [c for c in state.ada_casts if not c.scry]
    return casts[-1].damage if casts else 0


def last_hit_runes(state):
    for cast in reversed(state.ada_casts):
        if not cast.scry and cast.damage > 0:
            return cast.runes
    return None


def favourite_element(state):
    counts = Counter(e for c in state.ada_casts if not c.scry for e in c.elements)
    return counts.most_common(1)[0][0] if counts else None


def pick_shift(state, chart, rng):
    fav = favourite_element(state) or "Arcane"
    elements = sorted({d for (_, d) in chart if d not in ("Void", "Arcane")})
    resists = [e for e in elements if chart.get((fav, e), 1.0) < 1.0] or elements
    new = rng.choice(resists)
    beats = sorted(a for (a, d), m in chart.items() if d == new and m >= rules.SUPER_EFFECTIVE)
    return new, (beats[0] if beats else "Arcane")


def sync(conn, state):
    foe = state.foe
    if state.minions or foe.status.get("bats_made"):
        conn.execute("DELETE FROM enemies WHERE squad IS NOT NULL")
        conn.executemany(
            "INSERT INTO enemies (id, name, hp, max_hp, mp, max_mp, element, weakness, speed, squad) "
            "VALUES (?, ?, ?, ?, 0, 0, ?, ?, ?, ?)",
            [(b.status["id"], b.name, b.hp, b.max_hp, b.element, b.weakness, b.speed, b.squad)
             for b in living_bats(state)])
    if foe.status.get("hidden_hp"):
        conn.execute("UPDATE enemies SET hp = NULL, element = ?, weakness = ? WHERE id = ?",
                     (foe.element, foe.weakness, state.foe_id))
