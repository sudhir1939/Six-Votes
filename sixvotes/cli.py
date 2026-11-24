"""Command line entry point. Provided complete - you do not need to edit this file.

    python -m sixvotes audit   --predictions data/panel_predictions.csv --out outputs
    python -m sixvotes distill --digits data/digits.npz --out outputs
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sixvotes", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="leaderboard + panel independence audit")
    a.add_argument("--predictions", required=True, help="path to a predictions CSV")
    a.add_argument("--out", required=True, help="output folder")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--panel-size", type=int, default=3)

    d = sub.add_parser("distill", help="noise-distillation experiment")
    d.add_argument("--digits", required=True, help="path to digits.npz")
    d.add_argument("--out", required=True, help="output folder")

    args = parser.parse_args(argv)
    if args.command == "audit":
        from .io import load_predictions
        from .report import run_audit

        run_audit(load_predictions(args.predictions), args.out,
                  seed=args.seed, n_boot=args.n_boot, k=args.panel_size)
        print(f"wrote {args.out}/results.json and {args.out}/REPORT.md")
    else:
        from .distill import run

        run(args.digits, args.out)
        print(f"wrote {args.out}/subliminal_results.json")
    return 0
