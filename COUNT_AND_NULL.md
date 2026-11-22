# Count Aggregor and NULL: how they think

Guardians 5 and 7 of the design document (sections 5.7 and 5.9). They use the same turn loop as
the Slime and the Knight (`ai/guardian.py`): the brain picks an *intent*, a style writes it as
SQL, and only the query that really runs can hurt anyone.

Play them: `python main.py --start count` and `python main.py --start null`, or in the terminal
`python -m sim.play_cli --foe count` / `--foe null`. Press **F1** in the window (or type `:why`
in the terminal) to see everything the brain was thinking.

## AI techniques used

| # | Technique | Count Aggregor | NULL | File |
|---|---|---|---|---|
| 1 | Perception and memory | yes: HP, MP, threat, his swarm, a ledger of Ada's casts | yes: every cast's technique signature | `ai/perception.py`, `ai/count.py`, `ai/null.py` |
| 2 | State machine with `match`/`case` | 5 stances | 5 stances | `ai/count.py`, `ai/null.py` |
| 3 | Goal-driven plans | every tree leaf runs a plan; two-turn plans | Think picks a plan by utility | `ai/goals.py` |
| 4 | Steering behaviours | the Count | NULL | `ai/steering.py` |
| 5 | **Behaviour tree** + blackboard | main decision maker | | `ai/btree.py` |
| 6 | **Fuzzy logic** | how aggressive he feels | | `ai/fuzzy.py` |
| 7 | **Flocking** (boids) | the bats of each squad | | `ai/swarm.py` |
| 8 | **Agent state machines** | every bat: hover, dive, flee, regroup | | `ai/swarm.py` |
| 9 | **N-gram prediction** (N = 1..3, Witten-Bell) | | predicts Ada's next technique | `ai/player_model.py` |
| 10 | **Thompson sampling** (bandit learning) | | learns which counter works on this player | `ai/player_model.py` |
| 11 | **Lookahead search** (Monte Carlo, expectimax-style) | | simulates the next exchange for every move | `ai/null.py` |
| 12 | **Expected utility** decisions | utility selector for his finisher | utility for every move | `ai/btree.py`, `ai/null.py` |
| 13 | **AI Director** (dynamic difficulty) | scales his aggression | scales how much NULL explores | `ai/director.py` |
| 14 | **Barks** (rule-based reactive dialogue) | comments on what he noticed | says what it predicted | `ai/barks.py` |
| 15 | Seeded randomness | random selector, coin flips, cooldowns, squad sizes | Thompson samples, simulations | every battle replays from its seed |

---

## 1. Count Aggregor

### The fight
The Count hides behind squads of **Tally Bats**, real rows in `enemies` with a `squad` column.
While any squad has 3 or more bats (the **quorum**) nothing can hurt him. His book prints that
as `HAVING COUNT(*) >= 3`, which is how the player first reads HAVING. A row that names a squad
aims at it, and an area spell hits every bat in it. A JOIN of spells with twelve bats gives
one row per bat: the page overflows and repeated (spell, squad) pairs fizzle. `GROUP BY e.squad`
gives one row per squad, and `HAVING COUNT(*) >= 3` keeps only the squads that shield him.

Every query he writes has GROUP BY (so his first cast is the Invitation) and a HAVING that says,
in SQL, why the spell makes sense right now:

```sql
SELECT name FROM foe_spells WHERE foe_id = 10 AND kind = 'swarm'
GROUP BY name HAVING (SELECT COUNT(*) FROM enemies WHERE squad IS NOT NULL) >= 3
```

If the condition is false, the query returns nothing and his turn is wasted.

### His brain, one turn
1. **Perceive**: the shared senses plus his swarm: bats per squad, which squads hold the
   quorum, how many stragglers could regroup.
2. **Ledger**: he counts Ada too: which runes she uses, how often she groups, overflows, and how
   many bats each cast kills. If she groups, he goes **audited** and summons many small squads.
   If she doesn't, he *musters* bats into big squads that overflow her JOINs.
3. **Stance** (`match`): counting, recounting (quorum broken), gorging (badly hurt), frenzied
   (Ada weak, big swarm), audited.
4. **Fuzzy logic**: three fuzzy inputs (swarm few/some/lots, Ada's HP low/ok/high, his HP
   low/ok/high) and nine rules like *IF swarm is lots AND Ada is low THEN reckless* give one
   number, his **aggression** (careful 0.15, steady 0.5, reckless 0.85, averaged by rule strength).
5. **AI Director**: it multiplies that aggression by its intensity (more when Ada is winning,
   less when she struggles).
6. **Behaviour tree**, ticked from the root every turn (a higher branch can interrupt a running
   plan):

```
?Count's will                       ? selector   > sequence   ?~ random selector   ?$ utility
  >opening      first turn? -> [roll call]               (prints the quorum: the Invitation)
  >survive      badly hurt? has a swarm? 70% -> ?~mend [feed] [fortify]
  >reinforce    quorum broken? -> ?restore quorum
                    >stragglers  can regroup? -> [regroup: rally, then summon]   (2 turns)
                    cooldown(1) [summon]
  >punish       Ada overflowed or exhausted? swarm can bite? -> [pounce]
  >execute      Ada is weak? -> ?$finisher [swarm finish] [bolt finish]
  >expand       below his tally target? chance(more when careful) -> cooldown [expand]
  >guard        feels threatened? chance(more when careful) -> [guard]
  ?~harass      weights from the fuzzy aggression and the stance: [bite] [tithe] [drink]
  [rest]
```

   The summon cooldown is what gives Ada her window: right after a summon he can't rebuild a
   broken quorum by summoning, so she gets one turn to hit him.
7. **Goal plans**: each leaf runs a plan (move into position, then act).
8. **Flocking and agents**: every bat is an agent with its own state machine (hover near its
   post, **dive** at Ada when the swarm bites, **flee** when its squad is under the quorum,
   **regroup** after a Roll Call), and each squad flies as a boids flock.
9. **Barks**: he says what he noticed, e.g. *"You GROUP BY? Then I split my ranks."* when the
   ledger shows she groups, or *"Your page can't hold my army, witch."* when she overflows.

---

## 2. NULL, the Voice of Doubt

### The fight
NULL's hp, element and weakness are NULL in the table, so `=` never matches it and
`hp > 0` filters it out. Its **Veil of Doubt** only lets a spell through if the query looks at
the void: it reads `enemies` and uses `IS NULL` or `COALESCE`.

| Move | What it does | Beats |
|---|---|---|
| Fade to Grey | the technique that last hurt NULL passes through it from now on | players who repeat themselves |
| Shift of Nothing | takes an element that resists Ada's favourite, until it casts again | hard-coded elements (`'Arcane'`) |
| Whisper of Doubt | drains Ada's MP | big spenders, and healing loops |
| Echo of Doubt | throws back 60% of her last hit | players who hit hard |
| Empty Strike | damage | players who can't touch it |
| Hollow Mend | heals | when NULL is low |

Every NULL query tests for NULL, so its first cast teaches IS NULL. It writes in Ada's own
favourite style (JOIN, GROUP BY, LIMIT or WHERE).

### Its brain, one turn
1. **Learn**: each of Ada's casts becomes a *technique signature* (its runes, e.g.
   `COALESCE+IS NULL+JOIN+WHERE`). The N-gram model learns it, the last prediction is scored,
   and the bandit is told how well NULL's last move did against what she really did.
2. **Predict**: the N-gram counts which signature follows which 1 or 2 previous ones and mixes
   the orders with Witten-Bell interpolation. It trains on this fight and on every earlier fight
   of the run. Each predicted signature is sorted into a kind: **blind** (can't touch NULL),
   **repeat** (what last hurt it), **fixed** (IS NULL but no COALESCE), **adaptive** (COALESCE).
3. **Stance** (`match`): listening, mirroring, doubting, hunting, unravelling.
4. **Look ahead**: for each move it plays the next exchange 24 times: Ada's cast is drawn from
   the prediction, damage rolls are random. Score = share of her HP lost minus share of its own.
   A Fade is worth more than one turn, because it keeps blocking that technique, so the
   simulation counts the turns it will save too (this fixes the "horizon effect").
5. **Thompson sampling**: for each (move, kind) it keeps a Beta belief of how good that move is,
   starting from designer guesses. It samples from the beliefs, so moves it is unsure about
   still get tried, and moves that keep failing fade away.
6. **Utility** = 55% lookahead + 45% bandit belief, then the stance bias and a few situation
   rules (heal when hurt, drain her MP when her MP only buys a heal...). The AI Director decides
   how much it explores: when Ada leads, NULL trusts its belief averages more and plays sharper.
7. **Credit**: defensive moves are rewarded for damage she didn't do (unless that defence was
   already there), attacks for damage they did.
8. **Barks**: *"You will COALESCE again. I know it."*, *"Called it. 4 out of 5."*, or
   *"...I didn't see that coming."* when you surprise it.

---

## 3. Evaluation

`python -m sim.evaluate --n 300 --plots` plays seeded battles between scripted players and the
bosses through the real game (real SQL, rune gate, Rule of the Page). Full results:
`docs/eval/results.md`. Charts: `docs/eval/win_rates.png`, `docs/eval/null_learning_curve.png`.

| Boss | Player | Ada wins | Ada loses | Rounds |
|---|---|---|---|---|
| Count | brute: ignores the bats | 0% | 100% | 5.1 |
| Count | joiner: JOIN, no GROUP BY (overflows) | 0% | 100% | 5.1 |
| Count | grouper: GROUP BY, no HAVING | 89% | 2% | 9.9 |
| Count | expert: GROUP BY + HAVING | 100% | 0% | 10.2 |
| NULL | blind: never uses IS NULL | 0% | 100% | 8.9 |
| NULL | repetitive: the same good query every turn | 2% | 10% | 24.5 |
| NULL | alternating: two techniques in turn | 68% | 13% | 12.6 |
| NULL | varied: never the same way twice | 87% | 10% | 10.7 |

Each boss beats the player who ignores its lesson and loses to the one who uses it. (Fights
that are neither a win nor a loss hit the simulator's 25-round limit; the game has no limit.)

**Ablations** (same player, weaker brain):

| Brain | Player | Ada wins | Ada loses | Rounds | Ada HP left |
|---|---|---|---|---|---|
| Count, full | grouper | 89% | 2% | 9.9 | 62 |
| Count replaced by random choices | grouper | 99% | 0% | 5.8 | 86 |
| NULL, full | repetitive | **2%** | 10% | 24.5 | 48 |
| NULL without learning | repetitive | **58%** | 14% | 13.8 | 48 |
| NULL replaced by random choices | repetitive | 41% | 3% | 17.5 | 64 |
| NULL, full | varied | 87% | **10%** | 10.7 | 49 |
| NULL without lookahead | varied | 90% | **4%** | 13.4 | 55 |

- Learning is what makes NULL dangerous: switch it off and the repetitive player goes from 2% to 58% wins.
- The lookahead makes NULL deadlier against good players: Ada loses 2.5 times as often and fights are shorter.
- The behaviour tree makes the Count last almost twice as long as a random brain and costs Ada 24 more HP.

**Learning curve** (`null_learning_curve.png`, share of correct predictions so far): against
the repetitive player NULL is right ~60% of the time, against the alternating player it climbs
after turn 4 once the 3-gram sees the pattern, and against the varied player it stays below a
random guess. That's exactly why varied play wins.

Randomness: 11 to 19 different 4-move openings over 300 seeds in the real fights, 1.7 to 2.7
bits of move entropy, and every battle replays exactly from its seed. Each decision takes about
1 ms.

---

## 4. Files

| File | What it is |
|---|---|
| `ai/btree.py` | behaviour tree nodes: Selector, Sequence, RandomSelector, UtilitySelector, Inverter, Succeeder, Chance, Cooldown, Limit, Condition, Action, Blackboard |
| `ai/fuzzy.py` | fuzzy sets and rules |
| `ai/director.py` | the AI Director |
| `ai/barks.py` | the bark system |
| `ai/count.py` | Count Aggregor |
| `ai/swarm.py` | bat agents and flocking |
| `ai/player_model.py` | N-gram predictor, Thompson-sampling bandit |
| `ai/null.py` | NULL |
| `ai/styles.py` | how each boss writes its SQL |
| `magic/specials.py` | the rules: squads, quorum, summon, rally, swarm, feed; hidden HP, veil, fade, shift, doubt, echo |
| `engine/boss_art.py` | the art: barracks, Count, bats, the void, NULL |
| `sim/evaluate.py` | scripted players, numbers, charts |
| `tests/test_count_and_null.py` | 45 tests |

Boss ids are 10 (Count) and 20 (NULL), bats get ids from 100, so nothing clashes with the Twins.
