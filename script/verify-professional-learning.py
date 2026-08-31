#!/usr/bin/env python

import argparse
import json
from pathlib import Path
import sys

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))

from src.verification.professional_learning import (
    ProfessionalLearningCriteria,
    verify_professional_learning,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify held-out professional-learning telemetry against fixed thresholds",
    )
    parser.add_argument("telemetry_file")
    parser.add_argument("--minimum-cross-entropy-reduction", type=float, default=0.05)
    parser.add_argument("--minimum-top-one-gain", type=float, default=0.02)
    parser.add_argument("--minimum-top-five-gain", type=float, default=0.05)
    parser.add_argument("--maximum-value-mse-ratio", type=float, default=1.05)
    args = parser.parse_args()
    criteria = ProfessionalLearningCriteria(
        minimum_cross_entropy_reduction=args.minimum_cross_entropy_reduction,
        minimum_top_one_gain=args.minimum_top_one_gain,
        minimum_top_five_gain=args.minimum_top_five_gain,
        maximum_value_mse_ratio=args.maximum_value_mse_ratio,
    )
    report = verify_professional_learning(args.telemetry_file, criteria)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
