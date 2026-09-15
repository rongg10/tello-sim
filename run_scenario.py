#!/usr/bin/env python3
"""Run a scenario and produce the plots and video for it.

    python run_scenario.py scenarios/02_gust_during_move.yaml
    python run_scenario.py scenarios/04_swarm_formation.yaml --video
    python run_scenario.py --all --video

With --all this is everything in one command: every scenario flown, every log
written, a dashboard for each and a video to show.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tello_sim import render, scenarios


def run_one(path: Path, make_video: bool, fps: int, realtime: bool, verbose: bool) -> dict:
    started = time.perf_counter()
    simulator, swarm, recorder = scenarios.run(
        path, realtime=realtime, verbose=verbose
    )
    wall = time.perf_counter() - started

    result = {
        "scenario": path.name,
        "log": recorder.directory,
        "summary": simulator.summary(),
        "speedup": simulator.time / wall if wall > 0 else float("inf"),
    }

    result["dashboard"] = render.dashboard(recorder.directory)
    if make_video:
        result["video"] = render.animate(recorder.directory, fps=fps)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", type=Path)
    parser.add_argument("--all", action="store_true", help="run every scenario in scenarios/")
    parser.add_argument("--video", action="store_true", help="also render a flight video")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--realtime", action="store_true", help="run at wall-clock speed")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.all:
        paths = scenarios.list_scenarios()
    elif args.scenario:
        paths = [args.scenario]
    else:
        parser.error("give a scenario file, or --all")

    if not paths:
        print("No scenarios found in scenarios/")
        return 1

    results = []
    for path in paths:
        print(f"\n{'=' * 70}\n{path}\n{'=' * 70}")
        results.append(run_one(path, args.video, args.fps, args.realtime, not args.quiet))

    print(f"\n{'=' * 70}\nDone\n{'=' * 70}")
    for r in results:
        s = r["summary"]
        print(f"\n{r['scenario']}")
        print(f"  log        {r['log']}")
        print(f"  dashboard  {r['dashboard']}")
        if "video" in r:
            print(f"  video      {r['video']}")
        print(f"  flew       {s['sim_time']}s of simulated time at {r['speedup']:.0f}x")
        print(f"  battery    {s['battery_remaining']}")
        print(f"  collisions {s['collisions']}")
        if s["min_separation_m"] is not None:
            print(f"  closest two drones came: {s['min_separation_m']} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
