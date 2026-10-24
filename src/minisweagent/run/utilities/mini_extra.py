#!/usr/bin/env python3

"""This is the central entry point to the mini-extra script. Use subcommands
to invoke other command line utilities like running benchmarks and inspecting trajectories.
"""

import argparse
import sys
from importlib import import_module

SUBCOMMANDS = {
    "inspect": ("utilities.inspector", "Browse trajectories"),
    "swebench": ("benchmarks.swebench", "Run SWE-bench (batch mode)"),
    "swebench-single": ("benchmarks.swebench_single", "Run SWE-bench (single instance)"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run extra mini-SWE-agent commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in SUBCOMMANDS.items():
        subparsers.add_parser(name, help=description, add_help=False)
    args = sys.argv[1:] if argv is None else argv
    if not args:
        parser.print_help()
        return 0
    parsed, remaining = parser.parse_known_args(args)
    return import_module(f"minisweagent.run.{SUBCOMMANDS[parsed.command][0]}").app(
        remaining, prog=f"{parser.prog} {parsed.command}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
