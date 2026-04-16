from pathlib import Path
import sys


def ensure_scripts_dir_on_path() -> None:
    scripts_dir = Path(__file__).resolve().parents[1]
    scripts_dir_str = str(scripts_dir)

    if scripts_dir_str not in sys.path:
        sys.path.insert(0, scripts_dir_str)
