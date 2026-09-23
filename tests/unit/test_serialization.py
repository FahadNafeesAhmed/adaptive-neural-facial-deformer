"""Metadata interchange must preserve ordering and reject incompatible inputs."""

import json
import unittest

from anf_deformer.data import ControlSpec, RigSchema, dumps_rig_schema, loads_rig_schema


class RigSerializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = RigSchema(
            (ControlSpec("smile_left", -1.0, 1.0, 0.25), ControlSpec("jaw_open", 0, 1, 0)),
            vertex_count=2000,
        )
        self.document = json.loads(dumps_rig_schema(self.schema))

    def test_round_trip_preserves_control_order_and_values(self) -> None:
        restored = loads_rig_schema(dumps_rig_schema(self.schema))
        self.assertEqual(restored, self.schema)
        self.assertEqual(restored.neutral_values, (0.25, 0))

    def test_rejects_unsupported_or_missing_version(self) -> None:
        for version in (2, True, 1.0, "1", None):
            with self.subTest(version=version):
                self.document["schema_version"] = version
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    loads_rig_schema(json.dumps(self.document))
        del self.document["schema_version"]
        with self.assertRaisesRegex(ValueError, "missing"):
            loads_rig_schema(json.dumps(self.document))

    def test_rejects_unknown_fields_and_invalid_container_types(self) -> None:
        documents = [[], {**self.document, "controls": {}}, {**self.document, "typo": 1}]
        for document in documents:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    loads_rig_schema(json.dumps(document))

    def test_rejects_duplicate_json_fields(self) -> None:
        payload = dumps_rig_schema(self.schema).replace(
            '"vertex_count": 2000', '"vertex_count": 1, "vertex_count": 2000'
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
            loads_rig_schema(payload)

    def test_reuses_control_range_and_name_validation(self) -> None:
        self.document["controls"][0]["neutral"] = 2
        with self.assertRaisesRegex(ValueError, "neutral"):
            loads_rig_schema(json.dumps(self.document))
        self.document["controls"][0]["neutral"] = 0
        self.document["controls"][1]["name"] = "smile_left"
        with self.assertRaisesRegex(ValueError, "unique"):
            loads_rig_schema(json.dumps(self.document))

    def test_rejects_missing_control_fields_and_invalid_json(self) -> None:
        del self.document["controls"][0]["minimum"]
        with self.assertRaisesRegex(ValueError, "missing"):
            loads_rig_schema(json.dumps(self.document))
        with self.assertRaises(ValueError):
            loads_rig_schema("{broken json}")


if __name__ == "__main__":
    unittest.main()
