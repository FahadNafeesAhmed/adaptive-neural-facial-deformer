"""Pose samples preserve rig metadata ordering and mesh shape."""

import unittest

from anf_deformer.data import ControlSpec, PoseSample, RigSchema


class PoseSampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = RigSchema(
            controls=(ControlSpec("jaw_open", 0.0, 1.0, 0.0),),
            vertex_count=2,
        )

    def test_valid_pose_has_immutable_sequence_values(self) -> None:
        sample = PoseSample(self.schema, [0.5], [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

        self.assertEqual(sample.control_values, (0.5,))
        self.assertEqual(sample.vertices, ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0)))

    def test_rejects_wrong_control_count_or_range(self) -> None:
        vertices = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            PoseSample(self.schema, (), vertices)
        with self.assertRaises(ValueError):
            PoseSample(self.schema, (1.1,), vertices)

    def test_rejects_bad_mesh_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            PoseSample(self.schema, (0.5,), ((0.0, 0.0, 0.0),))
        with self.assertRaises(ValueError):
            PoseSample(self.schema, (0.5,), ((0.0, 0.0), (1.0, 0.0, 0.0)))
        with self.assertRaises(ValueError):
            PoseSample(self.schema, (0.5,), ((float("inf"), 0.0, 0.0), (1.0, 0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
