"""Small command-line checks for facial-deformer input metadata."""

import argparse
from pathlib import Path
import sys

from .data import loads_rig_schema


def main(argv: list[str] | None = None) -> int:
    """Validate rig metadata and return a shell-friendly exit status."""

    parser = argparse.ArgumentParser(prog="anf-deformer")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-rig", help="validate a rig metadata JSON file")
    validate.add_argument("path", type=Path, help="path to versioned rig metadata")
    args = parser.parse_args(argv)

    try:
        schema = loads_rig_schema(args.path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"Valid rig metadata: {len(schema.controls)} controls, "
        f"{schema.vertex_count} vertices, {schema.coordinate_space} coordinates."
    )
    return 0
