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

## License

[MIT](LICENSE)
