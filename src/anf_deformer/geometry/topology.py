"""Validated triangle connectivity shared across poses of one mesh."""

from dataclasses import dataclass

from anf_deformer.data.schema import RigSchema


@dataclass(frozen=True)
class MeshTopology:
    """Vertex count and ordered, zero-based triangle indices for a fixed mesh."""

    vertex_count: int
    triangles: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if isinstance(self.vertex_count, bool) or not isinstance(self.vertex_count, int) or self.vertex_count < 3:
            raise ValueError("vertex_count must be an integer of at least three")

        triangles = tuple(tuple(triangle) for triangle in self.triangles)
        if not triangles:
            raise ValueError("topology must contain at least one triangle")
        for triangle in triangles:
            if len(triangle) != 3:
                raise ValueError("each triangle must contain exactly three indices")
            if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.vertex_count for index in triangle):
                raise ValueError("triangle indices must be integers within vertex_count")
            if len(set(triangle)) != 3:
                raise ValueError("triangle indices must be distinct")

        object.__setattr__(self, "triangles", triangles)

    def validate_schema(self, schema: RigSchema) -> None:
        """Reject a rig whose vertex count differs from this topology."""

        if not isinstance(schema, RigSchema):
            raise TypeError("schema must be a RigSchema")
        if self.vertex_count != schema.vertex_count:
            raise ValueError("topology vertex count does not match the rig schema")
