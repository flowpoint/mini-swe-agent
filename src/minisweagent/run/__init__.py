"""CLI entry points and shared argument types."""

import argparse
import math


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not number >= 0:
        raise argparse.ArgumentTypeError("must be non-negative (or inf)")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer (0 means unlimited)")
    return number
