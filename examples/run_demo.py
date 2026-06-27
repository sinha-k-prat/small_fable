#!/usr/bin/env python3
"""run_demo.py — entry point for the residual_orrery demo.

Renders the side-by-side 0.5B vs 1.5B twin-sphere GIF for example 0 (the arithmetic
prompt) into out/. Runs on the pinned OLD stack (torch 2.2.2 / transformers 4.44.2 /
numpy 1.20.1 / mpl 3.2.2 / sklearn 0.23.1 / Pillow 7.2.0 / imageio 2.9.0), CPU/fp32.

    python examples/run_demo.py
    python examples/run_demo.py --example 2 --frames 60 --topk 48

This is a thin wrapper over ``python -m residual_orrery.cli`` — both paths are equivalent.
"""

import os
import sys

# make the package importable when run as a loose script from the repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from residual_orrery.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
