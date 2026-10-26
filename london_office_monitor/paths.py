"""Project locations, resolved from the package so commands work from any working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = PROJECT_ROOT / "data" / "seed"
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"
INTERNAL_DIR = PROJECT_ROOT / "data" / "internal"
SKILLS_DIR = PROJECT_ROOT / "skills"


def from_project(path: Path) -> Path:
    """An absolute path as given; a relative one is taken from the project root, not the current directory."""
    return path if path.is_absolute() else PROJECT_ROOT / path
