"""Utility modules for mini-SWE-agent (config management, inspector, etc.)."""

import argparse
import math


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number
