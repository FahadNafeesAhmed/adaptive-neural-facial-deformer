"""Topology validation protects fixed-mesh pose data from invalid indices."""

import unittest

from anf_deformer.data import ControlSpec, RigSchema
from anf_deformer.geometry import MeshTopology


class MeshTopologyTests(unittest.TestCase):
    def test_preserves_triangle_order_and_copies_input(self) -> None:
        source = [[0, 1, 2], [2, 1, 3]]
        topology = MeshTopology(4, source)
        source[0][0] = 3

        self.assertEqual(topology.triangles, ((0, 1, 2), (2, 1, 3)))

    def test_rejects_invalid_connectivity(self) -> None:
        for vertex_count, triangles in (
            (2, ((0, 1, 2),)),
            (3, ()),
            (3, ((0, 1),)),
            (3, ((0, 1, 3),)),
            (3, ((0, 1, 1),)),
            (3, ((0, True, 2),)),
        ):
            with self.subTest(vertex_count=vertex_count, triangles=triangles):
                with self.assertRaises(ValueError):
                    MeshTopology(vertex_count, triangles)

    def test_checks_rig_vertex_count(self) -> None:
        topology = MeshTopology(3, ((0, 1, 2),))
        controls = (ControlSpec("jaw_open", 0.0, 1.0, 0.0),)
        topology.validate_schema(RigSchema(controls, vertex_count=3))
        with self.assertRaises(ValueError):
            topology.validate_schema(RigSchema(controls, vertex_count=4))


if __name__ == "__main__":
    unittest.main()
