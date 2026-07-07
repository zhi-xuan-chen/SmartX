"""Generic retrieval manifest loading."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from downstream.retrieval.types import DatasetBuildResult, RetrievalCase

TEXT_COLUMNS = (
    "report_text",
    "text",
    "caption",
    "report",
    "Report",
    "findings",
    "Findings",
    "impression",
    "Impression",
)
IMAGE_COLUMNS = (
    "image_paths",
    "image_path",
    "image",
    "path",
    "filename",
    "file_name",
    "PngPath",
)
CASE_COLUMNS = ("case_uid", "uid", "id", "study_id", "StudyID")
STUDY_COLUMNS = ("study_id", "StudyID", "study")
SUBJECT_COLUMNS = ("subject_id", "PatientID", "patient_id")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("samples", "cases", "items", "data"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
        return pd.DataFrame(payload)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_list = list(columns)
    column_set = set(column_list)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    lower_to_original = {column.lower(): column for column in column_list}
    for candidate in candidates:
        match = lower_to_original.get(candidate.lower())
        if match:
            return match
    return None


def parse_sequence(value: object) -> list[object]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple)):
        return list(value)

    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    for separator in (";", "|"):
        if separator in text:
            return [item.strip() for item in text.split(separator) if item.strip()]
    return [value]


def parse_image_paths(value: object, image_root: str | Path | None = None) -> tuple[str, ...]:
    paths = []
    root = Path(image_root).expanduser() if image_root else None
    for item in parse_sequence(value):
        text = str(item).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if root is not None and not path.is_absolute():
            path = root / path
        paths.append(str(path))
    return tuple(paths)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _stable_subset_key(case: RetrievalCase, subset_seed: int) -> str:
    return hashlib.sha1(f"{subset_seed}:{case.case_uid}".encode("utf-8")).hexdigest()


def _limit_cases(
    cases: list[RetrievalCase],
    max_cases: int = 0,
    subset_seed: int = 0,
) -> tuple[RetrievalCase, ...]:
    if max_cases and max_cases > 0 and len(cases) > max_cases:
        return tuple(sorted(cases, key=lambda case: _stable_subset_key(case, subset_seed))[:max_cases])
    return tuple(cases)


def load_case_manifest(
    manifest_path: str | Path,
    *,
    dataset: str,
    split: str,
    image_root: str | Path | None = None,
    image_column: str | None = None,
    text_column: str | None = None,
    case_column: str | None = None,
    max_cases: int = 0,
    subset_seed: int = 0,
    verify_images: bool = False,
) -> DatasetBuildResult:
    manifest_path = Path(manifest_path)
    frame = read_table(manifest_path)
    image_col = image_column or pick_column(frame.columns, IMAGE_COLUMNS)
    text_col = text_column or pick_column(frame.columns, TEXT_COLUMNS)
    case_col = case_column or pick_column(frame.columns, CASE_COLUMNS)
    study_col = pick_column(frame.columns, STUDY_COLUMNS)
    subject_col = pick_column(frame.columns, SUBJECT_COLUMNS)
    split_col = pick_column(frame.columns, ("split", "Split", "partition"))

    if image_col is None:
        raise ValueError(f"Manifest has no image path column: {manifest_path}")
    if text_col is None:
        raise ValueError(f"Manifest has no report text column: {manifest_path}")

    cases: list[RetrievalCase] = []
    skipped_text = 0
    skipped_images = 0
    for idx, row in frame.iterrows():
        report_text = clean_text(row.get(text_col, ""))
        if not report_text:
            skipped_text += 1
            continue
        image_paths = parse_image_paths(row.get(image_col, ""), image_root)
        if not image_paths:
            skipped_images += 1
            continue
        if verify_images and any(not Path(path).exists() for path in image_paths):
            skipped_images += 1
            continue

        row_split = clean_text(row.get(split_col, split)) if split_col else split
        study_id = clean_text(row.get(study_col, "")) if study_col else ""
        subject_id = clean_text(row.get(subject_col, "")) if subject_col else ""
        case_uid = clean_text(row.get(case_col, "")) if case_col else ""
        if not case_uid:
            case_uid = f"{dataset}:{study_id or idx}"
        cases.append(
            RetrievalCase(
                case_uid=case_uid,
                dataset=dataset,
                split=row_split or split,
                image_paths=image_paths,
                report_text=report_text,
                study_id=study_id,
                subject_id=subject_id,
                metadata={"source_manifest": str(manifest_path)},
            )
        )

    limited = _limit_cases(cases, max_cases=max_cases, subset_seed=subset_seed)
    notes = [f"Loaded case manifest: {manifest_path}"]
    if len(limited) != len(cases):
        notes.append(f"Using deterministic subset: {len(limited)} of {len(cases)} cases.")
    return DatasetBuildResult(
        dataset=dataset,
        split=split,
        cases=limited,
        source_rows=len(frame),
        skipped_missing_text=skipped_text,
        skipped_missing_images=skipped_images,
        notes=tuple(notes),
    )


def case_manifest_rows(cases: Iterable[RetrievalCase]) -> list[dict[str, object]]:
    return [
        {
            "case_uid": case.case_uid,
            "dataset": case.dataset,
            "split": case.split,
            "study_id": case.study_id,
            "subject_id": case.subject_id,
            "num_images": len(case.image_paths),
            "image_paths": ";".join(case.image_paths),
            "report_text": case.report_text,
        }
        for case in cases
    ]
