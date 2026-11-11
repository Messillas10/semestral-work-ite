# Tests

## Approach

- **Plain `assert` functions**, no test framework: each test is a function `test_...` in `tests/`.
  `python test_main.py` runs all of them, prints PASS / FAIL per test and exits with code 1 if any fails.
- **Tiny hand-written dataset** (`tests/toy_data.csv`, 12 rows = 3 seasons x 4 fictional players) instead of
  the real 14 092-row CSV: every expected value can be checked by hand, and the tests do not depend on the real data.
- The toy rows are chosen to hit the edge cases: empty `3P%` (players who never shoot threes), a player with
  exactly half of the team's games, and a shortened 50-game season (2003).

## Sample input / output pairs

### `src/data.py`

| Function | Input | Expected output |
|---|---|---|
| `load_data` | `tests/toy_data.csv` | 12 rows, seasons 2001, 2002, 2003 |
| `load_data` | path that does not exist | `FileNotFoundError` |
| `clean_data` | toy data (4 empty `3P%`) | no empty `FG%` / `3P%` / `FT%` |
| `eligible_players` | toy data | 9 rows (20/82, 40/82 and 10/50 games removed); Epsilon Forward with exactly 41/82 kept |

### `src/model.py`

With only 2 candidates per toy season the model cannot learn anything meaningful, so the toy tests check
**properties** that must hold for any data. The quality of the model is checked by one **regression test** on the real
dataset: if a change makes the documented result worse, the test fails.

| Function | Input | Expected output |
|---|---|---|
| `predict_season` | toy data, season 2001 | only 2001, ranks 1, 2, 3, predicted share never negative |
| `predict_season` | toy data, season 1950 | `ValueError` |
| `predict_season` | toy data, season 2001, real `Share` of 2001 changed to 0.5 | same predictions (no leakage: the predicted season is not used for training) |
| `evaluate` + `top_k_hits` | real dataset (1991–2021) | top-1 **21/31**, top-3 **29/31** |

### `src/profiles.py`

| Function | Input | Expected output |
|---|---|---|
| `mark_mvps` | toy data | MVPs Alpha Center 2001, Beta Guard 2002, Alpha Center 2003 |
| `profile` | toy data | group sizes 3 MVPs, 3 other candidates, 9 league; mean `PTS` of MVPs 29.0 (= (30 + 28 + 29) / 3) |
| `find_clusters` | toy data, k = 2, run twice | clusters {0, 1}, identical both times (fixed `random_state`) |
| `find_clusters` + `describe_clusters` | real dataset, k = 4 | MVPs per cluster 0, 0, 11, 20 |

### `src/plots.py`

A chart cannot be compared with an `assert`: the tests check that a `Figure` with the expected number of bars is
returned. The look was checked by eye on the real data.

| Function | Input | Expected output |
|---|---|---|
| `season_chart` | toy prediction of 2001, top 2 | a `Figure` with 4 bars (2 players x predicted/real) |
| `accuracy_chart` | toy `evaluate` summary | 3 bars (one per season) |

### `src/cli.py`

`main([...])` is called with the toy CSV (`--data`) and a temporary `--output` folder.

| Command | Expected output |
|---|---|
| `predict 2001` | exit code 0, prints `Real MVP: Alpha Center` |
| `predict 1950` | exit code 1, prints `Error: Season 1950 not in data (...)` |
