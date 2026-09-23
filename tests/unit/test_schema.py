"""Checks for the metadata contract shared across pipeline stages."""

import unittest

from anf_deformer.data import ControlSpec, RigSchema


class RigSchemaTests(unittest.TestCase):
    def test_preserves_control_order_and_neutral_pose(self) -> None:
        jaw = ControlSpec("jaw_open", 0.0, 1.0, 0.0)
        smile = ControlSpec("smile_left", -1.0, 1.0, 0.25)
        schema = RigSchema([jaw, smile], vertex_count=2000)

        self.assertEqual(schema.controls, (jaw, smile))
        self.assertEqual(schema.neutral_values, (0.0, 0.25))
        self.assertEqual(schema.coordinate_space, "head_local")

    def test_rejects_invalid_control_ranges(self) -> None:
        with self.assertRaises(ValueError):
            ControlSpec("blink", 1.0, 0.0, 0.5)
        with self.assertRaises(ValueError):
            ControlSpec("blink", 0.0, 1.0, 2.0)
        with self.assertRaises(ValueError):
            ControlSpec("blink", 0.0, float("nan"), 0.0)

    def test_rejects_duplicate_names_and_invalid_mesh(self) -> None:
        control = ControlSpec("jaw_open", 0.0, 1.0, 0.0)
        with self.assertRaises(ValueError):
            RigSchema((control, control), vertex_count=2000)
        with self.assertRaises(ValueError):
            RigSchema((control,), vertex_count=0)
        with self.assertRaises(ValueError):
            RigSchema((control,), vertex_count=2000, coordinate_space="world")


if __name__ == "__main__":
    unittest.main()
