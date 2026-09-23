"""Explicit partition assignments for complete animation sequences."""

from dataclasses import dataclass
from typing import Literal

Partition = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SequenceSplit:
    """Keep all frames of each named source sequence in one partition."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for partition in ("train", "validation", "test"):
            values = getattr(self, partition)
            if isinstance(values, (str, bytes)):
                raise ValueError(f"{partition} must be a sequence of IDs, not a string")
            try:
                sequence_ids = tuple(values)
            except TypeError as error:
                raise ValueError(f"{partition} must be a sequence of IDs") from error
            if not sequence_ids:
                raise ValueError(f"{partition} must contain at least one sequence")
            for sequence_id in sequence_ids:
                if (
                    not isinstance(sequence_id, str)
                    or not sequence_id
                    or sequence_id != sequence_id.strip()
                ):
                    raise ValueError("sequence IDs must be nonempty strings without outer whitespace")
                if sequence_id in seen:
                    raise ValueError(f"sequence {sequence_id!r} is assigned more than once")
                seen.add(sequence_id)
            object.__setattr__(self, partition, sequence_ids)

    def partition_for(self, sequence_id: str) -> Partition:
        """Find an explicit assignment; unknown sequences never default to training."""

        if sequence_id in self.train:
            return "train"
        if sequence_id in self.validation:
            return "validation"
        if sequence_id in self.test:
            return "test"
        raise KeyError(f"unassigned sequence: {sequence_id!r}")
