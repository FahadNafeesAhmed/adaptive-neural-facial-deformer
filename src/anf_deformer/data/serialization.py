"""Versioned JSON interchange for rig metadata, without mesh payloads."""

import json
from dataclasses import asdict
from typing import Any

from .schema import ControlSpec, RigSchema

_SCHEMA_KEYS = {"schema_version", "controls", "vertex_count", "coordinate_space"}
_CONTROL_KEYS = {"name", "minimum", "maximum", "neutral"}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _require_keys(value: Any, expected: set[str], context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        raise ValueError(
            f"{context} fields do not match: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def dumps_rig_schema(schema: RigSchema) -> str:
    """Serialize metadata as readable JSON, preserving control order."""

    if not isinstance(schema, RigSchema):
        raise TypeError("schema must be a RigSchema")
    document = {"schema_version": 1, **asdict(schema)}
    return json.dumps(document, indent=2, allow_nan=False) + "\n"


def loads_rig_schema(payload: str) -> RigSchema:
    """Read version 1 metadata; reject ambiguous or unsupported documents.

    Malformed JSON, field mismatches, and invalid rig values raise ValueError.
    The controls array defines the order used by every pose sample.
    """

    document = json.loads(payload, object_pairs_hook=_unique_object)
    _require_keys(document, _SCHEMA_KEYS, "rig schema")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("unsupported schema_version; expected integer 1")
    if not isinstance(document["controls"], list):
        raise ValueError("controls must be a JSON array")

    controls = []
    for index, control in enumerate(document["controls"]):
        _require_keys(control, _CONTROL_KEYS, f"control {index}")
        controls.append(ControlSpec(**control))
    return RigSchema(
        controls=tuple(controls),
        vertex_count=document["vertex_count"],
        coordinate_space=document["coordinate_space"],
    )
