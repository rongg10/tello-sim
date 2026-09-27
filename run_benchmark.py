#!/usr/bin/env python3
"""Score scenarios instead of just flying them.

    python run_benchmark.py --all
    python run_benchmark.py scenarios/11_moving_door.yaml --seeds 10
    python run_benchmark.py --all --ablate-collisions
    python run_benchmark.py scenarios/08_slalom.yaml --wind 0 0.5 1.0 1.5

Every run writes a JSON file next to the summary, because the summary is what
you read today and the per-run records are what you need in six months when
somebody asks whether the difference was significant.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

from tello_sim import benchmark, scenarios


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run scenario benchmarks and report SR, OSR, SPL and energy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("scenario", nargs="*", help="scenario YAML files to score")
    parser.add_argument(
        "--all", action="store_true",
        help="score every scenario in --dir that has a benchmark block",
    )
    parser.add_argument("--dir", default="scenarios", help="where to look for scenarios")
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="how many seeds per scenario (default 5). One seed is an anecdote.",
    )
    parser.add_argument(
        "--seed-start", type=int, default=0, help="first seed (default 0)",
    )
    parser.add_argument(
        "--ablate-collisions", action="store_true",
        help="run each scenario twice, with contacts on and off, and compare",
    )
    parser.add_argument(
        "--wind", type=float, nargs="+", metavar="SPEED",
        help="sweep a steady headwind at these speeds in m/s, replacing the "
             "scenario's own wind",
    )
    parser.add_argument(
        "--no-collisions", action="store_true",
        help="run with contacts off: the drones become ghosts to everything "
             "except the floor",
    )
    parser.add_argument("--out", default="logs/benchmarks", help="where to write JSON")
    parser.add_argument("--quiet", action="store_true", help="summary lines only")
    return parser.parse_args(argv)


def collect(args) -> list:
    """Work out which scenarios to score, and complain usefully if none."""
    if args.all:
        found = benchmark.load_suite(args.dir)
        if not found:
            sys.exit(
                f"No scenario in {args.dir}/ has a benchmark block.\n"
                "Add one to a scenario file:\n"
                "  benchmark:\n"
                "    goal: [2.0, 1.0, 0.8]\n"
                "    success_radius: 0.5\n"
                "    time_limit: 60"
            )
        return found

    if not args.scenario:
        sys.exit("Name a scenario file, or pass --all. See --help.")

    out = []
    for path in args.scenario:
        scenario = scenarios.load(path)
        if not scenario.benchmark:
            sys.exit(
                f"{path} has no benchmark block, so there is nothing to score it "
                "against. Add a `benchmark:` section with a goal."
            )
        out.append(scenario)
    return out


def with_wind(scenario, speed: float):
    """A copy of a scenario flying into a steady headwind of a given speed."""
    clone = copy.deepcopy(scenario)
    clone.world = dict(clone.world)
    clone.world["wind"] = {"type": "constant", "velocity": [-float(speed), 0.0, 0.0]}
    return clone


def main(argv=None) -> int:
    args = parse_args(argv)
    suite = collect(args)
    seeds = range(args.seed_start, args.seed_start + args.seeds)

    out_dir = Path(args.out)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    reports = []

    for scenario in suite:
        if not args.quiet:
            print(f"\n{scenario.name}: {scenario.description.strip() or 'no description'}")

        if args.wind:
            for speed in args.wind:
                report = benchmark.run(
                    with_wind(scenario, speed), seeds=seeds,
                    label=f"{scenario.name} @ {speed:g} m/s",
                    verbose=not args.quiet,
                )
                reports.append(report)
                print(report.table())
        elif args.ablate_collisions:
            pair = benchmark.ablate_collisions(
                scenario, seeds=seeds, verbose=not args.quiet
            )
            reports.extend(pair)
            for report in pair:
                print(report.table())
        else:
            report = benchmark.run(
                scenario, seeds=seeds,
                collisions=False if args.no_collisions else None,
                verbose=not args.quiet,
            )
            reports.append(report)
            print(report.table())

    if len(reports) > 1:
        print("\n" + benchmark.compare(reports))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"benchmark-{stamp}.json"
    path.write_text(
        json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8"
    )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
