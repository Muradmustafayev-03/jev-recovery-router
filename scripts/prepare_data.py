"""Run from the checkout before or after editable installation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recovery_router.data import main

if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])
