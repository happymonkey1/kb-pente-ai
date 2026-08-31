#!/usr/bin/env python

from pathlib import Path
import sys

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))

from src.verification.tiny_learning import run_tiny_learning_verification


def main() -> int:
    report = run_tiny_learning_verification()
    print(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
