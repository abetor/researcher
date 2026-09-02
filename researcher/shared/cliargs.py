"""ArgumentParser with the workspace exit contract: usage errors exit with 1.

Standard argparse uses exit code 2 for malformed arguments, while this CLI reserves
1 for every usage or configuration failure. The subclass changes only the exit code;
usage and stderr text remain intact. Subparsers inherit the same behavior.
"""
from __future__ import annotations

import argparse
import sys


class ArgumentParser(argparse.ArgumentParser):
    """Behave like argparse.ArgumentParser but exit with 1 for usage errors."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")
