# NBA MVP Predictor

Analyses past NBA seasons to answer two questions:

1. **Who should have won the MVP?** — a model trained on the other seasons
   predicts the MVP vote share of every player and ranks the candidates.
2. **What kind of player wins the MVP?** — player profiles (clusters) and the
   statistics that separate MVP winners from the rest of the league.

> Work in progress — semestral work for the ITE course (CTU Prague).

Inputs, outputs and program logic (with UML diagrams): [docs/design.md](docs/design.md)

## Installation

Requirements: Python 3 (tested with 3.14) and Git. The dataset is included in `data/`.

```bash
git clone https://github.com/Messillas10/semestral-work-ite.git
cd semestral-work-ite
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies (`requirements.txt`): pandas (data), scikit-learn (regression and k-means), matplotlib (charts).

## Usage

```bash
python main.py predict 2016 --top 5   # MVP race of one past season (2016 = 2015-16)
python main.py evaluate               # predict every season 1991-2021, top-1 / top-3 accuracy
python main.py profile                # MVPs vs other candidates vs league + model coefficients
python main.py clusters --k 4         # player archetypes (k-means) and MVPs per archetype
python main.py --help                 # all options
```

Each command prints a text report and saves its chart as PNG in `output/` (change it with `--output DIR`;
use another CSV with `--data PATH`). Invalid input (unknown season, missing file) prints a short error and exits with code 1.

## Tests

```bash
python test_main.py
```

Runs every test in `tests/` and prints PASS / FAIL for each one. Approach and sample input/output pairs:
[docs/tests.md](docs/tests.md)

## License

[MIT](LICENSE)
