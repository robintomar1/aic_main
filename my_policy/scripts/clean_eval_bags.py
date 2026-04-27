#!/usr/bin/env python3
"""Continuously delete old eval-container bag recordings to keep disk usage bounded.

The eval container's scoring node writes a rosbag2 MCAP per trial into
/root/aic_results/ (bind-mounted to <docker>/results/ on host). Each bag is
~180 MB; a 500-trial run accumulates ~90 GB if nothing reaps them. Disabling
bag recording isn't supported (aic_scoring/src/ScoringTier2.cc writes-then-
reads the bag for scoring), but bags become reapable as soon as the per-trial
scoring step has finished — no policy or recorder reads them downstream.

This script polls the results directory and deletes bag dirs whose most
recent file mtime is older than --max-age-seconds (default: 300 = 5 min,
generous margin over the typical sub-minute scoring step).

ONLY deletes directories matching the pattern `bag_trial_*` — `scoring.yaml`
and any other non-bag artifacts are left untouched.

Run on host (faster than running inside the dev container; bags appear
immediately in the bind-mounted directory):

    python3 my_policy/scripts/clean_eval_bags.py \\
        --results-dir /home/robin/ssd/aic_workspace/aic_docker/aic/results

Stop with Ctrl-C.

Why not just delete everything periodically? --keep-recent N preserves the N
most-recent bags so you can always inspect the last few trials if scoring
flagged something weird. Default keep-recent is 2 (penultimate + current).
"""

from __future__ import annotations

import argparse
import shutil
import signal
import sys
import time
from pathlib import Path

_BAG_PATTERN = "bag_trial_*"


def _bag_mtime(bag_dir: Path) -> float:
    """Most recent mtime across the bag's contents. We check files INSIDE
    the dir (not the dir itself) because a bag dir's mtime updates only on
    file create/delete, not on file content writes — so an actively-being-
    written bag could look "stale" by directory mtime alone."""
    latest = bag_dir.stat().st_mtime
    try:
        for child in bag_dir.iterdir():
            if child.is_file():
                latest = max(latest, child.stat().st_mtime)
    except FileNotFoundError:
        # Race: dir vanished between glob and iterdir. Treat as ancient so
        # caller doesn't try to delete it again.
        return 0.0
    return latest


def _scan_bags(results_dir: Path) -> list[tuple[Path, float]]:
    """Returns list of (bag_dir, latest_mtime_inside) sorted oldest-first."""
    bags = []
    for bag in results_dir.glob(_BAG_PATTERN):
        if not bag.is_dir():
            continue
        bags.append((bag, _bag_mtime(bag)))
    bags.sort(key=lambda b: b[1])
    return bags


def _delete(bag_dir: Path, dry_run: bool) -> int:
    """Returns bytes freed (or estimated, if dry_run)."""
    try:
        size = sum(f.stat().st_size for f in bag_dir.rglob("*") if f.is_file())
    except FileNotFoundError:
        return 0
    if dry_run:
        return size
    try:
        shutil.rmtree(bag_dir)
    except FileNotFoundError:
        return 0
    return size


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results-dir", type=Path,
        default=Path("/home/robin/ssd/aic_workspace/aic_docker/aic/results"),
        help="Directory containing bag_trial_* subdirs.",
    )
    p.add_argument(
        "--max-age-seconds", type=float, default=300.0,
        help="Delete bags whose most-recent file mtime is older than this. "
             "Default: 300 (5 min). Generous margin over per-trial scoring.",
    )
    p.add_argument(
        "--keep-recent", type=int, default=2,
        help="Always keep at least this many most-recent bags regardless of age. "
             "Default: 2.",
    )
    p.add_argument(
        "--poll-interval-seconds", type=float, default=30.0,
        help="How often to scan the directory.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be deleted without deleting.",
    )
    args = p.parse_args()

    if not args.results_dir.exists():
        print(f"error: results dir not found: {args.results_dir}", file=sys.stderr)
        return 2
    if args.max_age_seconds <= 0:
        print("error: --max-age-seconds must be > 0", file=sys.stderr)
        return 2
    if args.keep_recent < 0:
        print("error: --keep-recent must be >= 0", file=sys.stderr)
        return 2

    print(f"watching {args.results_dir}")
    print(f"  pattern: {_BAG_PATTERN}/")
    print(f"  max age: {args.max_age_seconds:.0f}s")
    print(f"  keep recent: {args.keep_recent}")
    print(f"  poll: {args.poll_interval_seconds:.0f}s")
    print(f"  dry run: {args.dry_run}")
    print(f"Ctrl-C to stop.\n")

    stop = False

    def _on_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    total_deleted = 0
    total_bytes = 0
    while not stop:
        try:
            bags = _scan_bags(args.results_dir)
        except FileNotFoundError:
            time.sleep(args.poll_interval_seconds)
            continue
        now = time.time()
        # Bags are sorted oldest-first; keep the last N regardless of age.
        eligible = bags[: max(0, len(bags) - args.keep_recent)]
        deleted_this_pass = 0
        bytes_this_pass = 0
        for bag_dir, mtime in eligible:
            age = now - mtime
            if age < args.max_age_seconds:
                continue
            size = _delete(bag_dir, args.dry_run)
            verb = "would delete" if args.dry_run else "deleted"
            print(f"  {verb} {bag_dir.name} (age={age:.0f}s, size={size / 1e6:.0f} MB)")
            deleted_this_pass += 1
            bytes_this_pass += size
        if deleted_this_pass:
            total_deleted += deleted_this_pass
            total_bytes += bytes_this_pass
            print(f"  pass: {deleted_this_pass} bags, {bytes_this_pass / 1e9:.2f} GB; "
                  f"total: {total_deleted} bags, {total_bytes / 1e9:.2f} GB")
        # Interruptible sleep.
        for _ in range(int(args.poll_interval_seconds * 10)):
            if stop:
                break
            time.sleep(0.1)

    print(f"\nstopped. total: {total_deleted} bags, {total_bytes / 1e9:.2f} GB freed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
