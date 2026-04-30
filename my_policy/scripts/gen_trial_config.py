#!/usr/bin/env python3
"""Thin shim — the actual implementation moved to my_policy.gen_trial_config.

Kept here so:
  - `pixi run python my_policy/scripts/gen_trial_config.py ...` still works
    as the documented entry point in the README;
  - host-runnable tests in this directory that do `import gen_trial_config as
    gtc` (after sys.path-inserting their own dir) keep working without churn.

Source-of-truth lives in the installed package so package-side modules
(my_policy/localizer/labels.py, etc.) can import it regardless of install
layout. Don't add logic here — edit `my_policy/my_policy/gen_trial_config.py`.
"""
from __future__ import annotations

# Re-export every public symbol so `import gen_trial_config as gtc` is
# indistinguishable from importing the package module directly.
from my_policy.gen_trial_config import *  # noqa: F401,F403
from my_policy.gen_trial_config import main  # explicit so __main__ works


if __name__ == "__main__":
    main()
