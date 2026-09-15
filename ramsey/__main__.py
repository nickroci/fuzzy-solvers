"""CLI entry point for the Ramsey catalogue audits."""

from __future__ import annotations

import sys

from ramsey.catalogue import main

if __name__ == "__main__":
    sys.exit(main())
