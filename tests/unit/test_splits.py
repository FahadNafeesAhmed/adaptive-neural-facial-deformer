"""Sequence assignments must keep held-out clips out of training."""

import unittest

from anf_deformer.data import SequenceSplit


class SequenceSplitTests(unittest.TestCase):
    def test_assignments_preserve_order_and_copy_mutable_inputs(self) -> None:
        train = ["smile_take_02", "smile_take_01"]
        split = SequenceSplit(train, ["blink_take_01"], ["speech_take_01"])
        train.append("speech_take_01")
        self.assertEqual(split.train, ("smile_take_02", "smile_take_01"))
        self.assertEqual(split.partition_for("smile_take_01"), "train")
        self.assertEqual(split.partition_for("blink_take_01"), "validation")
        self.assertEqual(split.partition_for("speech_take_01"), "test")

    def test_rejects_overlap_between_any_pair_of_partitions(self) -> None:
        cases = [
            (("a",), ("a",), ("c",)),
            (("a",), ("b",), ("a",)),
            (("a",), ("b",), ("b",)),
        ]
        for partitions in cases:
            with self.subTest(partitions=partitions):
                with self.assertRaisesRegex(ValueError, "assigned more than once"):
                    SequenceSplit(*partitions)

    def test_rejects_duplicates_inside_a_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, "assigned more than once"):
            SequenceSplit(("a", "a"), ("b",), ("c",))

    def test_rejects_missing_partitions_and_invalid_ids(self) -> None:
        for train in ((), "clip_a", None, ("",), (" padded ",), (42,)):
            with self.subTest(train=train):
                with self.assertRaises(ValueError):
                    SequenceSplit(train, ("b",), ("c",))
        with self.assertRaises(ValueError):
            SequenceSplit(("a",), (), ("c",))
        with self.assertRaises(ValueError):
            SequenceSplit(("a",), ("b",), ())

    def test_unassigned_sequence_is_an_error(self) -> None:
        split = SequenceSplit(("a",), ("b",), ("c",))
        with self.assertRaises(KeyError):
            split.partition_for("new_clip")


if __name__ == "__main__":
    unittest.main()
