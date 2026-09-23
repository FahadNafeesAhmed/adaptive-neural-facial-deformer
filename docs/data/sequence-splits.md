# Sequence partitions

`SequenceSplit` records explicit training, validation, and test assignments for complete animation sequences. It requires a nonempty collection of unique sequence IDs in each partition and rejects overlap. Unknown sequences raise `KeyError` when queried; they never silently become training data.

```python
from anf_deformer.data import SequenceSplit

split = SequenceSplit(
    train=("smile_take_01", "jaw_take_01"),
    validation=("blink_take_01",),
    test=("speech_take_01",),
)
assert split.partition_for("speech_take_01") == "test"
```

All frames from a source sequence must use that sequence's assignment. IDs are case-sensitive and must be stable across re-exports, derived versions, and frame extraction. Related material should use a common source-group ID when holding out individual clips would still leak information.

This class validates assignments, not the content of a dataset. It cannot detect duplicate geometry under different IDs, verify that an exporter used the correct ID, or prevent another component from loading test data. Those checks belong in the future data loader. Training and adaptive selection must respect these assignments and leave test sequences untouched.

Assignments are supplied by the caller. This contract does not generate random splits, evaluate a model, or claim that any particular split measures generalization well.
