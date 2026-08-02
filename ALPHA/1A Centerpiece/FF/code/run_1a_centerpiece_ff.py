from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ALPHA.alpha_common import evaluate_centerpiece


if __name__ == "__main__":
    evaluate_centerpiece("FF", Path(__file__).resolve().parents[1])
