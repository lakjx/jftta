from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.experiments import paired_main

if __name__ == "__main__":
    paired_main("direct_backup")
