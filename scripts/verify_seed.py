"""Check every automatically verifiable seed figure against its source page.

Usage: uv run python scripts/verify_seed.py data/seed
"""
import sys

from london_office_monitor.seed.verify import main

if __name__ == "__main__":
    sys.exit(main())
