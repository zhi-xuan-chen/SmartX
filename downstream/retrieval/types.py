"""Retrieval data containers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalCase:
    case_uid: str
    dataset: str
    split: str
    image_paths: tuple[str, ...]
    report_text: str
    study_id: str = ""
    subject_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetBuildResult:
    dataset: str
    split: str
    cases: tuple[RetrievalCase, ...]
    source_rows: int
    skipped_missing_text: int
    skipped_missing_images: int
    notes: tuple[str, ...] = ()
