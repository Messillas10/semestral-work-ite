"""Entry point of the NBA MVP predictor.

Usage examples:
    python main.py predict 2016 --top 5
    python main.py evaluate
    python main.py profile
    python main.py clusters --k 4
    python main.py --help
"""

import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())  # EXIT CODE 0 = OK, 1 = ERROR
