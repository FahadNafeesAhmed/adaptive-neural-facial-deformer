# Rig metadata JSON

`dumps_rig_schema` and `loads_rig_schema` exchange the current `RigSchema` contract using JSON strings. They perform no file writes and require only the Python standard library.

```python
from anf_deformer.data import ControlSpec, RigSchema, dumps_rig_schema, loads_rig_schema

schema = RigSchema((ControlSpec("jaw_open", 0.0, 1.0, 0.0),), vertex_count=2000)
payload = dumps_rig_schema(schema)
restored = loads_rig_schema(payload)
assert restored == schema
```

The version 1 document contains exactly:

- `schema_version`: integer `1`.
- `controls`: a nonempty ordered array of objects with `name`, `minimum`, `maximum`, and `neutral`.
- `vertex_count`: a positive integer.
- `coordinate_space`: `"head_local"`.

Control array order is significant: pose values use this same order. Object field order is not significant. Duplicate JSON fields, unknown or missing fields, unsupported versions, and invalid rig values are rejected with `ValueError`.

This is an initial metadata format, not a complete dataset manifest. It does not store mesh arrays, units, topology identity, asset provenance, or train/test assignments. These remain requirements for the future exporter; vertex count alone cannot establish topology compatibility.
