"""Validated pose input and reference mesh for one fixed-topology rig."""

from dataclasses import dataclass
from math import isfinite

from .schema import RigSchema


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


@dataclass(frozen=True)
class PoseSample:
    """A rig pose and its evaluated head-local mesh in schema order."""

    schema: RigSchema
    control_values: tuple[float, ...]
    vertices: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema, RigSchema):
            raise TypeError("schema must be a RigSchema")

        control_values = tuple(self.control_values)
        if len(control_values) != len(self.schema.controls):
            raise ValueError("control value count does not match the rig schema")
        for value, control in zip(control_values, self.schema.controls):
            if not _finite_number(value) or not control.minimum <= value <= control.maximum:
                raise ValueError(f"invalid value for control {control.name!r}")

        vertices = tuple(tuple(vertex) for vertex in self.vertices)
        if len(vertices) != self.schema.vertex_count:
            raise ValueError("vertex count does not match the rig schema")
        if any(len(vertex) != 3 or not all(_finite_number(value) for value in vertex) for vertex in vertices):
            raise ValueError("vertices must contain finite 3D coordinates")

        object.__setattr__(self, "control_values", control_values)
        object.__setattr__(self, "vertices", vertices)
