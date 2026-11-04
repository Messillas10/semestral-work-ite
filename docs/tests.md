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
| `load_data` | CSV without the `Share` column | `ValueError` whose message names `Share` |
| `clean_data` | toy data (4 empty `3P%`) | no empty `FG%` / `3P%` / `FT%`; Alpha Center 2001 has `3P%` = 0 |
| `clean_data` | toy data | the input DataFrame still has its 4 empty values (works on a copy) |
| `eligible_players` | toy data | 9 rows; Delta Bench 2001 (20 of 82 games), Zeta Rookie 2002 (40 of 82) and Delta Bench 2003 (10 of 50) removed |
| `eligible_players` | Epsilon Forward 2002: 41 games, team 41-41 | kept (the limit `G >= (W + L) / 2` is inclusive) |
| `eligible_players` | 2003, 50-game season | Gamma Wing (30 games) kept, Delta Bench (10 games) removed |

### `src/model.py`

With only 2 candidates per toy season the model cannot learn anything meaningful, so the toy tests check
**properties** that must hold for any data. The quality of the model is checked by one **regression test** on the real
dataset: if a change makes the documented result worse, the test fails.

| Function | Input | Expected output |
|---|---|---|
| `predict_season` | toy data, season 2001 | 3 rows (eligible players of 2001), ranks 1, 2, 3, predicted share sorted and never negative |
| `predict_season` | toy data, season 1950 | `ValueError` whose message names 1950 |
| `predict_season` | toy data, season 2001, real `Share` of 2001 changed to 0.5 | same ranking and predictions (no leakage: the predicted season is not used for training) |
| `predict_season` | toy data, season 2001, `PTS` of Gamma Wing 2003 (no votes) changed to 99 | same predictions (the model trains on vote receivers only) |
| `evaluate` | toy data | one summary row per season (2001, 2002, 2003), real MVPs Alpha Center, Beta Guard, Alpha Center; 9 predictions (one per eligible player) |
| `top_k_hits` | real-MVP ranks [1, 2, 5] | k = 1 → 1, k = 3 → 2, k = 10 → 3 |
| `coefficients` | toy data | one weight per feature, sorted by absolute value |
| `evaluate` + `coefficients` | real dataset (1991–2021) | top-1 **21/31**, top-3 **29/31**, Stephen Curry rank 1 in 2016, strongest features PTS and W/L% |
