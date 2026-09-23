"""Validated metadata for a single character's facial controls and mesh."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ControlSpec:
    """One ordered rig control with its valid range and neutral value."""

    name: str
    minimum: float
    maximum: float
    neutral: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("control name must be nonempty and have no outer whitespace")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
            for value in (self.minimum, self.maximum, self.neutral)
        ):
            raise ValueError(f"control {self.name!r} values must be finite numbers")
        if self.minimum >= self.maximum:
            raise ValueError(f"control {self.name!r} minimum must be below maximum")
        if not self.minimum <= self.neutral <= self.maximum:
            raise ValueError(f"control {self.name!r} neutral must be within its range")


@dataclass(frozen=True)
class RigSchema:
    """Metadata needed to interpret a fixed-topology facial training sample."""

    controls: tuple[ControlSpec, ...]
    vertex_count: int
    coordinate_space: str = "head_local"

    def __post_init__(self) -> None:
        controls = tuple(self.controls)
        if not controls or any(not isinstance(control, ControlSpec) for control in controls):
            raise ValueError("controls must contain ControlSpec values")
        names = [control.name for control in controls]
        if len(names) != len(set(names)):
            raise ValueError("control names must be unique")
        if isinstance(self.vertex_count, bool) or not isinstance(self.vertex_count, int) or self.vertex_count <= 0:
            raise ValueError("vertex_count must be a positive integer")
        if self.coordinate_space != "head_local":
            raise ValueError("only head_local coordinates are supported")
        object.__setattr__(self, "controls", controls)

    @property
    def neutral_values(self) -> tuple[float, ...]:
        """Return neutral control values in the schema's fixed order."""

        return tuple(control.neutral for control in self.controls)
