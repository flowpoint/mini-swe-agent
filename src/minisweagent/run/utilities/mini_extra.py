#!/usr/bin/env python3

"""Entry point for optional CLI utilities."""

import argparse
import sys
from importlib import import_module

SUBCOMMANDS = {
    "inspect": ("utilities.inspector", "Browse trajectories"),
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
