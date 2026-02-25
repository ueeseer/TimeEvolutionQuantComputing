from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support direct script execution: `python GroupProject/run_time_evolution_project.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from GroupProject.time_evolution.project_pipeline import RunConfig, run_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full PH10110 time-evolution project pipeline."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("GroupProject/results"),
        help="Directory for figures, data and metrics.",
    )
    parser.add_argument("--t-max", type=float, default=6.0)
    parser.add_argument("--n-times", type=int, default=121)
    parser.add_argument("--noise-px", type=float, default=0.002)
    parser.add_argument("--noise-pz", type=float, default=0.006)
    parser.add_argument("--trajectories", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260211)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke run for iteration; reduces runtime and fidelity.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.quick:
        cfg = RunConfig(
            t_max=min(args.t_max, 3.0),
            n_times=min(args.n_times, 61),
            trotter_order_for_plots=2,
            noise_p_x=args.noise_px,
            noise_p_z=args.noise_pz,
            noise_trajectories=min(args.trajectories, 8),
            seed=args.seed,
            error_steps=(12, 24, 48, 96),
        )
    else:
        cfg = RunConfig(
            t_max=args.t_max,
            n_times=args.n_times,
            trotter_order_for_plots=2,
            noise_p_x=args.noise_px,
            noise_p_z=args.noise_pz,
            noise_trajectories=args.trajectories,
            seed=args.seed,
            error_steps=(20, 40, 80, 160),
        )

    summary = run_project(output_dir=args.output_dir, cfg=cfg)
    print(f"Completed {len(summary['cases'])} project cases.")
    print(f"Results written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
