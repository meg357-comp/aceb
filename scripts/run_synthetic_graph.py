from __future__ import annotations

import argparse
from pathlib import Path

from cmveb.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible graph synthetic experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic_graph_small.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("results/synthetic_graph"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    overrides = {"overwrite": True} if args.overwrite else None
    run_dir = run_experiment(
        experiment_type="synthetic_graph",
        config_path=args.config,
        output_root=args.output_root,
        overrides=overrides,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
