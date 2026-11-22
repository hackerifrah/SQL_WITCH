# Evaluation: Count Aggregor and NULL (300 seeded battles per row)

## Count Aggregor (behaviour tree + fuzzy logic)

| Guardian | Player | Ada wins | Ada loses | Rounds | Ada HP left | Move entropy (bits) | Distinct openings | ms/turn |
|---|---|---|---|---|---|---|---|---|
| count | brute | 0% | 100% | 5.1 | 0 | 1.53 | 27 | 0.72 |
| count | joiner | 0% | 100% | 5.1 | 0 | 0.72 | 1 | 0.67 |
| count | grouper | 89% | 2% | 9.9 | 62 | 2.72 | 11 | 0.62 |
| count | expert | 100% | 0% | 10.2 | 63 | 2.65 | 17 | 0.64 |

Moves against the expert: summon 30%, bolt 22%, feed 13%, muster 12%, roll_call 10%, rally 9%, ward 3%, swarm 2%

## NULL (N-gram prediction + lookahead + Thompson sampling)

| Guardian | Player | Ada wins | Ada loses | Rounds | Ada HP left | Move entropy (bits) | Distinct openings | ms/turn | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| null | blind | 0% | 100% | 8.9 | 0 | 1.14 | 13 | 0.74 | 79% |
| null | repetitive | 2% | 10% | 24.5 | 48 | 1.68 | 12 | 0.90 | 58% |
| null | alternating | 68% | 13% | 12.6 | 48 | 2.13 | 16 | 0.93 | 46% |
| null | varied | 87% | 10% | 10.7 | 49 | 2.11 | 19 | 0.72 | 17% |

`accuracy`: how often NULL's top prediction of Ada's next technique was right.

## Ablations

The same players against weaker versions of each brain. A real brain should beat the random baseline, and NULL should be stronger with learning and with a longer context than without.

| Guardian | Player | Ada wins | Ada loses | Rounds | Ada HP left | Move entropy (bits) | Distinct openings | ms/turn | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| count | expert | 100% | 0% | 10.2 | 63 | 2.65 | 17 | 0.64 |  |
| count (random baseline) | expert | 100% | 0% | 5.6 | 85 | 3.17 | 292 | 0.39 |  |
| count | grouper | 89% | 2% | 9.9 | 62 | 2.72 | 11 | 0.62 |  |
| count (random baseline) | grouper | 99% | 0% | 5.8 | 86 | 3.17 | 295 | 0.48 |  |
| null | repetitive | 2% | 10% | 24.5 | 48 | 1.68 | 12 | 0.90 | 58% |
| null (random baseline) | repetitive | 41% | 3% | 17.5 | 64 | 2.81 | 282 | 0.42 |  |
| null (no learning) | repetitive | 58% | 14% | 13.8 | 48 | 2.20 | 19 | 0.94 | 67% |
| null (1-gram only) | repetitive | 2% | 14% | 24.2 | 47 | 1.70 | 16 | 0.93 | 58% |
| null | varied | 87% | 10% | 10.7 | 49 | 2.11 | 19 | 0.78 | 17% |
| null (no lookahead) | varied | 90% | 4% | 13.4 | 55 | 2.06 | 71 | 0.63 | 20% |

- NULL learning curve vs repetitive: 1: 100%, 2: 99%, 3: 100%, 4: 91%, 5: 36%, 6: 22%, 7: 17%, 8: 38%, 9: 47%, 10: 62%
- NULL learning curve vs alternating: 1: 100%, 2: 0%, 3: 0%, 4: 28%, 5: 42%, 6: 59%, 7: 76%, 8: 16%, 9: 76%, 10: 58%
- NULL learning curve vs varied: 1: 24%, 2: 19%, 3: 19%, 4: 9%, 5: 12%, 6: 22%, 7: 18%, 8: 14%, 9: 21%, 10: 14%
