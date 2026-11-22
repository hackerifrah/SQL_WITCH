"""Count Aggregor (behaviour tree, Tally Bats) and NULL (player model): rules, brains, balance."""

import random

import pytest

from ai import btree
from ai.base import Intent
from ai.btree import Blackboard, Status
from ai.count import CountBrain, make_plan
from ai.null import NullBrain, classify
from ai.player_model import CounterBandit, NGram, entropy, signature
from ai.roster import GUARDIANS, ORDER, next_guardian
from ai.steering import Kinematic, Vec
from ai.styles import render_count, render_null
from ai.swarm import Flock, next_state, squad_posts
from magic import guard, specials
from magic.battle import Battle
from magic.cast import resolve
from magic.rules import CastMemory

GROUPED = ("SELECT s.name, e.squad FROM spells AS s JOIN enemies AS e ON s.element = e.weakness "
           "WHERE s.name = 'Holy Flash' GROUP BY e.squad")
UNGROUPED = ("SELECT s.name, e.squad FROM spells AS s JOIN enemies AS e ON s.element = e.weakness "
             "WHERE s.kind = 'area'")
VOID = ("SELECT s.name, e.name FROM spells AS s JOIN enemies AS e "
        "ON s.element = COALESCE(e.weakness, 'Arcane') WHERE e.hp IS NULL AND s.power = 22")


def count_battle(seed=1, **kw):
    b = Battle(seed=seed, guardian="count", foe_opens=False, **kw)
    b.state.ada.runes |= {"GROUP BY", "HAVING", "DISTINCT"}
    return b


def null_battle(seed=1, **kw):
    b = Battle(seed=seed, guardian="null", foe_opens=False, **kw)
    b.state.ada.runes |= {"IS NULL", "COALESCE", "GROUP BY"}
    b.turn_order, b.queue = ["ada", "foe"], ["ada", "foe"]    # NULL is faster: let Ada start
    return b


# ------------------------------------------------------------------ the story

def test_both_guardians_come_after_the_knight_in_the_story():
    assert ORDER.index("count") > ORDER.index("knight")
    assert next_guardian("knight") in ("count", "twins")
    assert next_guardian("count") == "null" and next_guardian("null") is None


def test_the_invitations_hand_over_a_family_of_runes():
    b = Battle(seed=3, guardian="count")
    turn = b.foe_act()
    assert turn.invitation == "GROUP BY"
    assert {"GROUP BY", "HAVING", "DISTINCT"} <= b.state.ada.runes
    n = Battle(seed=3, guardian="null")
    assert n.foe_act().invitation == "IS NULL"
    assert {"IS NULL", "COALESCE"} <= n.state.ada.runes


# ------------------------------------------------------------------ behaviour trees

def bb(seed=0):
    return Blackboard(random.Random(seed), turn=1)


def test_selector_takes_the_first_child_that_does_not_fail():
    order = []
    tree = btree.Selector("s",
                          btree.Action("a", lambda b: order.append("a") or False),
                          btree.Action("b", lambda b: order.append("b") or True),
                          btree.Action("c", lambda b: order.append("c") or True))
    assert tree.tick(bb()) is Status.SUCCESS and order == ["a", "b"]


def test_sequence_stops_at_the_first_failure_and_pauses_on_running():
    seq = btree.Sequence("q", btree.Condition("yes", lambda b: True),
                         btree.Action("run", lambda b: Status.RUNNING),
                         btree.Action("never", lambda b: 1 / 0))
    assert seq.tick(bb()) is Status.RUNNING
    assert btree.Sequence("f", btree.Condition("no", lambda b: False),
                          btree.Action("never", lambda b: 1 / 0)).tick(bb()) is Status.FAILURE


def test_decorators():
    ok = btree.Action("ok", lambda b: True)
    assert btree.Inverter("not", ok).tick(bb()) is Status.FAILURE
    assert btree.Chance("never", 0.0, ok).tick(bb()) is Status.FAILURE
    assert btree.Chance("always", 1.0, ok).tick(bb()) is Status.SUCCESS
    board = bb()
    cool = btree.Cooldown("cd", 2, ok)
    assert cool.tick(board) is Status.SUCCESS
    for turn in (2, 3):
        board.turn = turn
        assert cool.tick(board) is Status.FAILURE           # cooling down
    board.turn = 4
    assert cool.tick(board) is Status.SUCCESS


def test_random_selector_is_random_but_reproducible():
    def first(seed):
        picked = []
        tree = btree.RandomSelector("r", [1, 1, 1], *[
            btree.Action(n, lambda b, n=n: picked.append(n) or True) for n in "abc"])
        tree.tick(bb(seed))
        return picked[0]
    firsts = {first(s) for s in range(30)}
    assert firsts == {"a", "b", "c"}
    assert first(7) == first(7)


def test_the_trace_shows_the_path_that_decided():
    tree = btree.Selector("root", btree.Sequence("no", btree.Condition("c", lambda b: True),
                                                 btree.Condition("d", lambda b: False)),
                          btree.Action("yes", lambda b: True))
    board = bb()
    tree.tick(board)
    assert btree.success_path(board.trace) == ["?root", "yes"]     # 'c' succeeded in a failed branch
    assert btree.tried_branches(board.trace) == ["x >no", "+ yes"]


# ------------------------------------------------------------------ the swarm and the quorum

def test_the_count_opens_with_two_squads_that_hold_the_quorum():
    b = count_battle()
    squads = specials.squads(b.state)
    assert len(squads) == 2 and all(3 <= len(v) <= 5 for v in squads.values())
    rows = b.conn.execute("SELECT squad, COUNT(*) FROM enemies WHERE squad IS NOT NULL "
                          "GROUP BY squad HAVING COUNT(*) >= 3").fetchall()
    assert len(rows) == 2 and specials.quorum(b.state)


def test_the_quorum_shields_the_count():
    b = count_battle()
    report = b.ada_cast("SELECT name FROM spells WHERE element = 'Light' AND kind = 'attack' "
                        "AND power = 34")
    assert b.state.foe.hp == b.state.foe.max_hp
    assert any(e[0] == "say" and "quorum" in e[1] for e in report.events)


def test_group_by_collapses_the_swarm_into_one_row_per_squad():
    b = count_battle()
    bats = len(specials.living_bats(b.state))
    # The Count's own row (squad NULL) is a group too, and that spell bounces off his quorum...
    plain = count_battle().ada_cast(GROUPED)
    assert len(plain.rows) == 3 and (None in [r[1] for r in plain.rows])
    # ...which is what HAVING is for: only the squads that hold the quorum.
    grouped = b.ada_cast(GROUPED + " HAVING COUNT(*) >= 3")
    assert len(grouped.rows) == 2 and grouped.overflow == 0
    assert len([c for c in grouped.casts if c.spell == "Holy Flash"]) == bats   # every bat hit


def test_without_group_by_the_join_overflows_and_drains_the_page():
    b = count_battle()
    report = b.ada_cast(UNGROUPED)
    assert len(report.rows) > 4 and report.overflow > 0
    assert report.mp_after == 0 or report.mp_spent > 30


def test_a_squad_falls_and_the_quorum_breaks_then_the_count_can_be_hurt():
    b = count_battle()
    for bat in b.state.minions:
        bat.hp = 1
    b.ada_cast(GROUPED + " HAVING COUNT(*) >= 3")
    assert not specials.living_bats(b.state) and not specials.quorum(b.state)
    assert b.conn.execute("SELECT COUNT(*) FROM enemies WHERE squad IS NOT NULL").fetchone() == (0,)
    b.foe_act()                                       # whatever he does, then Ada strikes
    if not specials.quorum(b.state):
        b.ada_cast("SELECT name FROM spells WHERE element = 'Light' AND kind = 'attack' "
                   "AND power = 34")
        assert b.state.foe.hp < b.state.foe.max_hp


def test_summon_muster_and_rally():
    b = count_battle()
    s = b.state
    s.minions.clear()
    squad, n = specials.summon_squad(s, 3, split=True)
    squad2, _ = specials.summon_squad(s, 2, split=True)
    assert squad != squad2
    specials.summon_squad(s, 1, split=False)          # muster: tops up the smallest squad
    assert sorted(len(v) for v in specials.squads(s).values()) == [3, 3]
    for bat in list(specials.squads(s)[squad])[:2]:
        bat.hp = 0
    s.minions = [m for m in s.minions if m.alive]
    keep, moved = specials.rally(s)                   # 1 + 3? no: only weak squads regroup
    assert moved in (0, 1)


def test_every_count_query_is_one_select_he_may_write_and_returns_one_spell():
    b = count_battle()
    for action in ("bolt", "summon", "muster", "rally", "swarm", "feed", "ward", "roll_call"):
        sql = render_count(Intent(action), b.state, 1)
        assert guard.inspect(sql, b.state.foe.runes) >= {"GROUP BY"}
        rows = b.conn.execute(sql).fetchall()
        if action == "rally":
            assert len(rows) == 1                     # two squads: the gate is true
        elif action == "roll_call":
            assert all(r[1] >= 3 for r in rows)       # the quorum, as a HAVING
        else:
            assert len(rows) == 1


def test_a_false_having_wastes_his_turn():
    b = count_battle()
    b.state.minions.clear()
    specials.sync(b.conn, b.state)
    sql = render_count(Intent("swarm"), b.state, 1)
    report = resolve(b.conn, b.state, sql, "foe", b.rng, b.chart)
    assert report.scry and b.state.ada.hp == b.state.ada.max_hp


# ------------------------------------------------------------------ the Count's brain

def test_the_count_opens_with_a_roll_call_and_explains_his_tree():
    b = Battle(seed=2, guardian="count")
    turn = b.foe_act()
    assert turn.reasoning["intent"] == "roll_call"
    assert "opening" in turn.reasoning["tree"]
    assert turn.reasoning["swarm"].startswith(str(len(b.state.minions)))


def test_the_count_rebuilds_a_broken_quorum():
    b = count_battle()
    b.state.minions.clear()
    specials.sync(b.conn, b.state)
    brain = b.brain
    brain.turns = 5                                   # past the opening
    intent = brain.decide(b.state, random.Random(0))
    assert intent.action in ("summon", "muster")
    assert "reinforce" in brain.explain()["tree"]


def test_the_count_pounces_when_ada_overflows():
    b = count_battle()
    b.ada_cast(UNGROUPED)                             # her page overflows
    picks = set()
    for s in range(5):
        brain = CountBrain()
        brain.turns = 5                               # past the opening roll call
        picks.add(brain.decide(b.state, random.Random(s)).action)
    assert "swarm" in picks


def test_the_audit_splits_the_swarm_against_an_aggregating_witch():
    b = count_battle()
    brain = b.brain
    for turn in (1, 2, 3):
        b.state.ada_casts.append(CastMemory(turn, ("GROUP BY", "JOIN", "WHERE"), ("Light",), 50,
                                            True, 0, False))
    brain.turns = 5
    brain.decide(b.state, random.Random(0))
    assert brain.mode == "audited" and brain.ledger.group_rate == 1.0


def test_a_running_plan_is_interrupted_by_a_higher_branch():
    b = count_battle()
    brain = b.brain
    plan = make_plan("regroup", ["rally", "summon"], brain)
    plan.status = plan.status.__class__.ACTIVE
    brain.plans["regroup"] = plan
    brain.turns = 0                                   # next turn is 1: the opening wins
    brain.decide(b.state, random.Random(0))
    assert "regroup" not in brain.plans


def test_the_count_is_not_deterministic():
    moves = set()
    for seed in range(20):
        b = Battle(seed=seed, guardian="count")
        seq = []
        while b.whose_turn != "over" and len(seq) < 5:
            if b.whose_turn == "foe":
                seq.append(b.foe_act().reasoning["intent"])
            else:
                b.ada_cast("SELECT name FROM spells WHERE kind = 'heal' AND power = 30 "
                           "AND element = 'Arcane'")
        moves.add(tuple(seq))
    assert len(moves) > 1


# ------------------------------------------------------------------ flocking

def test_bats_flock_by_squad_and_keep_apart():
    rng = random.Random(1)
    anchors = squad_posts(["Left Wing", "Right Wing"], Vec(250, 45), False, Vec(40, 130))
    flock = Flock()
    flock.sync([(f"b{i}", "Left Wing" if i < 4 else "Right Wing") for i in range(8)], anchors, rng)
    for frame in range(240):
        flock.update(anchors, 1 / 60, frame * 17)
    for bat in flock.bats.values():
        assert (bat.body.position - anchors[bat.squad]).length() < 25
    left = [b for b in flock.bats.values() if b.squad == "Left Wing"]
    assert all((a.body.position - b.body.position).length() > 0.5
               for a in left for b in left if a is not b)
    flock.sync([("b0", "Right Wing")], anchors, rng)           # rally + losses
    assert set(flock.bats) == {"b0"} and flock.bats["b0"].squad == "Right Wing"


def test_squads_fly_at_ada_when_the_swarm_bites():
    home = squad_posts(["Left Wing"], Vec(250, 45), False, Vec(40, 130))["Left Wing"]
    bite = squad_posts(["Left Wing"], Vec(250, 45), True, Vec(40, 130))["Left Wing"]
    assert (bite - Vec(40, 130)).length() < (home - Vec(40, 130)).length()


# ------------------------------------------------------------------ NULL's rules

def test_null_hides_its_values_as_null():
    b = null_battle()
    assert b.conn.execute("SELECT hp, element, weakness FROM enemies").fetchall() == [(None, None, None)]
    assert b.conn.execute("SELECT COUNT(*) FROM enemies WHERE hp > 0").fetchone() == (0,)
    b.ada_cast(VOID)
    assert b.state.foe.hp < b.state.foe.max_hp
    assert b.conn.execute("SELECT hp FROM enemies").fetchone() == (None,)    # still NULL


def test_the_veil_lets_through_only_queries_that_look_at_the_void():
    b = null_battle()
    b.ada_cast("SELECT name FROM spells WHERE element = 'Arcane' AND power = 22")
    assert b.state.foe.hp == b.state.foe.max_hp
    b2 = null_battle()
    b2.ada_cast(VOID)
    assert b2.state.foe.hp < b2.state.foe.max_hp


def test_fade_blocks_the_technique_that_last_hurt_it_but_not_another():
    b = null_battle()
    b.ada_cast(VOID)
    hp = b.state.foe.hp
    resolve(b.conn, b.state, render_null(Intent("fade"), b.state, 1), "foe", b.rng, b.chart)
    b._advance()                                       # NULL's turn is over
    assert b.state.foe.status["fade"]
    b.ada_cast(VOID)
    assert b.state.foe.hp == hp                        # the same signature passes through
    b.queue = ["ada"]                                  # Ada again (skip NULL's turn)
    b.ada_cast(VOID + " GROUP BY s.name")               # a different technique lands
    assert b.state.foe.hp < hp


def test_shift_resists_her_favourite_element_until_null_casts_again():
    b = null_battle()
    b.ada_cast(VOID)
    resolve(b.conn, b.state, render_null(Intent("shift"), b.state, 1), "foe", b.rng, b.chart)
    element, weakness = b.state.foe.element, b.state.foe.weakness
    assert element and weakness and b.chart[("Arcane", element)] <= 1.0
    assert b.conn.execute("SELECT element FROM enemies").fetchone() == (element,)
    resolve(b.conn, b.state, render_null(Intent("strike"), b.state, 2), "foe", b.rng, b.chart)
    assert b.state.foe.element is None                  # the void again


def test_doubt_drains_mp():
    b = null_battle()
    before = b.state.ada.mp
    resolve(b.conn, b.state, render_null(Intent("doubt"), b.state, 1), "foe", b.rng, b.chart)
    assert b.state.ada.mp < before


def test_null_writes_in_adas_style_and_always_tests_for_null():
    b = null_battle()
    sql = render_null(Intent("strike"), b.state, 1)
    assert "IS NULL" in guard.inspect(sql, b.state.foe.runes)
    for t in range(3):
        b.state.ada_casts.append(CastMemory(t, ("JOIN", "WHERE"), (), 0, False, 0, False))
    joined = render_null(Intent("strike"), b.state, 2)
    assert "JOIN" in guard.inspect(joined, b.state.foe.runes)
    assert b.conn.execute(joined).fetchall() == [("Empty Strike", "Ada")]


# ------------------------------------------------------------------ the player model

def test_signatures_and_classes():
    assert signature(("WHERE", "JOIN")) == "JOIN+WHERE" and signature(()) == "plain"
    assert classify("LIMIT+WHERE", None) == "blind"
    assert classify("COALESCE+IS NULL+JOIN", None) == "adaptive"
    assert classify("IS NULL+JOIN", "IS NULL+JOIN") == "repeat"
    assert classify("IS NULL+WHERE", None) == "fixed"


def test_the_ngram_learns_a_pattern_and_stays_a_distribution():
    model = NGram(3)
    seq = ["A", "B", "A", "B", "A", "B", "A"]
    model.train(seq)
    dist = model.predict(seq, ["A", "B", "C"])
    assert abs(sum(dist.values()) - 1) < 1e-9
    assert max(dist, key=dist.get) == "B" and dist["C"] > 0          # smoothed
    assert model.order_used(seq) == 3
    assert entropy(dist) < entropy({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})


def test_the_bandit_moves_its_belief_with_rewards():
    bandit = CounterBandit({("fade", "repeat"): (1, 1)})
    for _ in range(10):
        bandit.update("fade", "repeat", 1.0)
    assert bandit.mean("fade", "repeat") > 0.85 and bandit.learned("fade", "repeat") > 0.3
    rng = random.Random(0)
    assert 0 <= bandit.sample("fade", "repeat", rng) <= 1


def test_null_predicts_a_repetitive_player_and_fades():
    b = null_battle(seed=4)
    seen_fade = False
    for _ in range(6):
        if b.whose_turn == "ada":
            b.ada_cast(VOID)
        if b.whose_turn == "foe":
            seen_fade |= b.foe_act().reasoning["intent"] == "fade"
    assert seen_fade
    assert b.brain.hits >= 2


def test_null_studies_earlier_fights():
    brain = NullBrain()
    brain.observe_run([("LIMIT",), ("LIMIT",), ("LIMIT",), ("LIMIT",)])
    b = null_battle()
    brain.decide(b.state, random.Random(0))
    assert brain.top_guess == "LIMIT" and brain.mode != "listening"
    battle = Battle(seed=1, guardian="null", history=[("WHERE",)] * 5)
    assert battle.brain.old_history == ["WHERE"] * 5


def test_same_seed_same_battle():
    def run(seed):
        b = Battle(seed=seed, guardian="null")
        log = []
        for _ in range(6):
            if b.whose_turn == "over":
                break
            if b.whose_turn == "foe":
                log.append(b.foe_act().sql)
            else:
                b.ada_cast(VOID.replace("22", "60"))
        return log, b.state.foe.hp
    assert run(9) == run(9)


# ------------------------------------------------------------------ balance (design doc 9.1)

@pytest.mark.parametrize("guardian, weak, strong", [("count", "brute", "expert"),
                                                     ("null", "blind", "varied")])
def test_each_guardian_beats_who_ignores_its_lesson_and_loses_to_who_uses_it(guardian, weak, strong):
    from sim.evaluate import run
    assert run(guardian, weak, 15)["win"] == 0.0
    assert run(guardian, strong, 15)["win"] >= 0.8


def test_null_punishes_repetition_and_learning_is_what_does_it():
    from sim.evaluate import run
    repetitive = run("null", "repetitive", 20)["win"]
    varied = run("null", "varied", 20)["win"]
    no_learning = run("null", "repetitive", 20, make_brain=lambda: NullBrain(learning=False))["win"]
    assert repetitive < varied
    assert repetitive < no_learning


# ------------------------------------------------------------------ fuzzy logic, director, barks

def test_fuzzy_sets_and_rules():
    from ai.fuzzy import left_shoulder, right_shoulder, triangle
    assert triangle(5, 0, 5, 10) == 1.0 and triangle(2.5, 0, 5, 10) == 0.5 and triangle(11, 0, 5, 10) == 0
    assert left_shoulder(1, 2, 4) == 1.0 and left_shoulder(3, 2, 4) == 0.5
    assert right_shoulder(5, 2, 4) == 1.0 and right_shoulder(3, 2, 4) == 0.5


def test_the_count_is_reckless_with_a_big_swarm_and_careful_when_alone_and_hurt():
    brain = CountBrain()
    big = brain.fuzzy.run(swarm=12, ada=0.2, me=0.9)
    small = brain.fuzzy.run(swarm=1, ada=0.95, me=0.2)
    assert big > 0.7 and small < 0.3


def test_the_director_sharpens_when_ada_leads_and_eases_when_she_struggles():
    from ai.director import HIGHEST, LOWEST, Director
    d = Director()
    for _ in range(10):
        d.update(1.0, 0.2)
    assert d.intensity > 1.2 and d.intensity <= HIGHEST
    for _ in range(20):
        d.update(0.1, 1.0)
    assert d.intensity < 0.8 and d.intensity >= LOWEST


def test_barks_pick_the_top_rule_and_rest_it():
    from ai.barks import Bark, Barker
    barker = Barker([Bark("hi", lambda i: i["x"] > 1, ("big {x}",), 5),
                     Bark("idle", lambda i: True, ("idle",), 1)])
    rng = random.Random(0)
    assert barker.speak({"x": 3}, 1, rng) == "big 3"
    assert barker.speak({"x": 3}, 2, rng) == "idle"            # 'hi' is cooling down
    assert barker.speak({"x": 0}, 9, rng) == "idle"


def test_bosses_say_something_about_what_they_perceive():
    b = Battle(seed=5, guardian="count")
    assert b.foe_act().reasoning["says"]
    n = null_battle(seed=5)
    n.ada_cast(VOID)
    turn = n.foe_act()
    assert turn.reasoning["says"] and len(turn.reasoning["says"]) <= 44


def test_bats_have_states():
    flock = Flock()
    flock.sync([("a", "Left Wing"), ("b", "Left Wing")], {"Left Wing": Vec(200, 40)}, random.Random(0))
    bat = flock.bats["a"]
    assert next_state(bat, 2, False) == "flee"          # under the quorum
    assert next_state(bat, 5, True) == "dive"
    flock.sync([("a", "Right Wing")], {"Right Wing": Vec(220, 40)}, random.Random(0))
    assert flock.bats["a"].state == "regroup"


def test_null_looks_ahead_and_fades_a_faded_player_never():
    b = null_battle(seed=2)
    b.ada_cast(VOID)
    b.foe_act()
    brain = b.brain
    assert set(brain.sim_values) >= {"strike", "doubt"}
    assert "lookahead" in brain.explain()
