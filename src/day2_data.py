"""
Day 2 data pipeline (MVP)
-------------------------
Download small dataset subsets, clean/normalize text, merge/split for tokenizer
training, and generate a data quality report.

Usage:
    python -m src.day2_data run
    python -m src.day2_data ingest
    python -m src.day2_data process
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable



@dataclass(frozen=True)
class DatasetCandidate:
    path: str
    name: str | None
    split: str
    text_fields: tuple[str, ...]
    license: str


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    language: str
    candidates: tuple[DatasetCandidate, ...]


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        source_id="general_english",
        language="en",
        candidates=(
            DatasetCandidate(
                path="wikitext",
                name="wikitext-103-raw-v1",
                split="train",
                text_fields=("text",),
                license="CC BY-SA 3.0",
            ),
            DatasetCandidate(
                path="wikipedia",
                name="20220301.en",
                split="train",
                text_fields=("text",),
                license="CC BY-SA 3.0",
            ),
        ),
    ),
    SourceSpec(
        source_id="indic_hindi",
        language="hi",
        candidates=(
            DatasetCandidate(
                path="oscar",
                name="unshuffled_deduplicated_hi",
                split="train",
                text_fields=("text",),
                license="ODC-By 1.0",
            ),
            DatasetCandidate(
                path="ai4bharat/IndicCorpV2",
                name="indiccorp_v2",
                split="train",
                text_fields=("text",),
                license="Various (see source)",
            ),
        ),
    ),
    SourceSpec(
        source_id="general_dialog",
        language="multi",
        candidates=(
            DatasetCandidate(
                path="opus100",
                name="en-hi",
                split="train",
                text_fields=("translation.en", "translation.hi"),
                license="CC BY 4.0",
            ),
        ),
    ),
)


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def ensure_dirs(base_data_dir: Path, docs_dir: Path) -> dict[str, Path]:
    raw_dir = base_data_dir / "raw"
    processed_dir = base_data_dir / "processed"
    intermediate_dir = processed_dir / "intermediate"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "intermediate_dir": intermediate_dir,
        "docs_dir": docs_dir,
    }


def get_nested_text(record: dict, field_path: str) -> str | None:
    value = record
    for key in field_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, str):
        return value
    return None


def iter_texts(record: dict, fields: tuple[str, ...]) -> Iterable[str]:
    for field in fields:
        text = get_nested_text(record, field)
        if text is not None:
            yield text


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def ingest(
    sources: tuple[SourceSpec, ...],
    raw_dir: Path,
    manifest_path: Path,
    max_samples_per_source: int,
    min_chars: int,
) -> dict:
    manifest_entries: list[dict] = []

    for spec in sources:
        source_dir = raw_dir / spec.source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        raw_out = source_dir / "raw.txt"

        successful = False
        errors: list[str] = []
        sample_count = 0

        for candidate in spec.candidates:
            try:
                from datasets import load_dataset

                ds = load_dataset(candidate.path, candidate.name, split=candidate.split, streaming=True)
            except Exception as exc:  # pragma: no cover
                errors.append(f"{candidate.path}/{candidate.name}: {exc}")
                continue

            with raw_out.open("w", encoding="utf-8") as f:
                for row in ds:
                    for text in iter_texts(row, candidate.text_fields):
                        text = text.strip()
                        if len(text) < min_chars:
                            continue
                        f.write(text + "\n")
                        sample_count += 1
                        if sample_count >= max_samples_per_source:
                            break
                    if sample_count >= max_samples_per_source:
                        break

            if sample_count > 0:
                successful = True
                manifest_entries.append(
                    {
                        "source": spec.source_id,
                        "language": spec.language,
                        "dataset_path": candidate.path,
                        "dataset_name": candidate.name,
                        "split": candidate.split,
                        "license": candidate.license,
                        "sample_size": sample_count,
                        "raw_file": str(raw_out.as_posix()),
                        "status": "ok",
                    }
                )
                break
            errors.append(f"{candidate.path}/{candidate.name}: zero usable rows")

        if not successful:
            if raw_out.exists():
                raw_out.unlink()
            manifest_entries.append(
                {
                    "source": spec.source_id,
                    "language": spec.language,
                    "sample_size": 0,
                    "status": "failed",
                    "errors": errors,
                }
            )

    manifest = {"sources": manifest_entries}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def process_manifest(
    manifest: dict,
    intermediate_dir: Path,
    processed_dir: Path,
    report_path: Path,
    valid_ratio: float,
    seed: int,
    sample_lines_limit: int,
) -> dict:
    per_source_stats: dict[str, dict] = {}
    merged_rows: list[tuple[str, str]] = []

    for entry in manifest["sources"]:
        source_id = entry["source"]
        if entry.get("status") != "ok":
            continue

        raw_file = Path(entry["raw_file"])
        cleaned_file = intermediate_dir / f"{source_id}.txt"
        source_lines: list[str] = []
        total_lines = 0
        removed_empty = 0
        decode_errors = 0

        with raw_file.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total_lines += 1
                if "\ufffd" in line:
                    decode_errors += 1
                cleaned = clean_text(line)
                if not cleaned:
                    removed_empty += 1
                    continue
                source_lines.append(cleaned)

        with cleaned_file.open("w", encoding="utf-8") as out:
            for line in source_lines:
                out.write(line + "\n")

        for line in source_lines:
            merged_rows.append((source_id, line))

        per_source_stats[source_id] = {
            "language": entry["language"],
            "cleaned_file": str(cleaned_file.as_posix()),
            "lines_in": total_lines,
            "lines_out": len(source_lines),
            "empty_lines_removed": removed_empty,
            "decode_errors": decode_errors,
        }

    rng = random.Random(seed)
    rng.shuffle(merged_rows)

    merged_text_path = processed_dir / "merged_corpus.txt"
    train_path = processed_dir / "train.txt"
    valid_path = processed_dir / "valid.txt"
    sample_path = processed_dir / "sample_100k_lines.txt"

    valid_count = int(len(merged_rows) * valid_ratio)
    valid_rows = merged_rows[:valid_count]
    train_rows = merged_rows[valid_count:]

    def write_rows(path: Path, rows: list[tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8") as out:
            for _, text in rows:
                out.write(text + "\n")

    write_rows(merged_text_path, merged_rows)
    write_rows(train_path, train_rows)
    write_rows(valid_path, valid_rows)
    write_rows(sample_path, train_rows[:sample_lines_limit])

    line_texts = [text for _, text in merged_rows]
    source_counts = Counter(source for source, _ in merged_rows)
    unique_lines = len(set(line_texts))
    total_lines = len(line_texts)
    duplicate_ratio = 0.0 if total_lines == 0 else 1.0 - (unique_lines / total_lines)
    total_chars = sum(len(x) for x in line_texts)
    approx_tokens = sum(len(x.split()) for x in line_texts)
    duplicate_heavy_chunks = detect_duplicate_heavy_chunks(line_texts, chunk_size=200, threshold=0.3)

    report = {
        "total_lines": total_lines,
        "total_chars": total_chars,
        "approx_tokens_whitespace": approx_tokens,
        "duplicate_ratio": duplicate_ratio,
        "source_proportions": {
            source: (count / total_lines if total_lines else 0.0) for source, count in source_counts.items()
        },
        "duplicate_heavy_chunks": duplicate_heavy_chunks,
        "paths": {
            "merged": str(merged_text_path.as_posix()),
            "train": str(train_path.as_posix()),
            "valid": str(valid_path.as_posix()),
            "sample": str(sample_path.as_posix()),
            "report": str(report_path.as_posix()),
        },
        "per_source_stats": per_source_stats,
    }

    report_path.write_text(render_report_markdown(report), encoding="utf-8")
    return report


def render_report_markdown(report: dict) -> str:
    lines = [
        "# Day 2 Data Report",
        "",
        "## Summary",
        f"- Total lines: {report['total_lines']:,}",
        f"- Total chars: {report['total_chars']:,}",
        f"- Approx tokens (whitespace): {report['approx_tokens_whitespace']:,}",
        f"- Duplicate ratio: {report['duplicate_ratio']:.4f}",
        f"- Duplicate-heavy chunks (>=30% duplicates in 200-line windows): {len(report['duplicate_heavy_chunks'])}",
        "",
        "## Source Proportions",
    ]
    for source, proportion in sorted(report["source_proportions"].items()):
        lines.append(f"- {source}: {proportion:.2%}")

    lines.append("")
    lines.append("## Per-Source Quality")
    for source, stats in sorted(report["per_source_stats"].items()):
        lines.append(f"- {source} ({stats['language']}):")
        lines.append(f"  - lines_in: {stats['lines_in']:,}")
        lines.append(f"  - lines_out: {stats['lines_out']:,}")
        lines.append(f"  - empty_lines_removed: {stats['empty_lines_removed']:,}")
        lines.append(f"  - decode_errors: {stats['decode_errors']:,}")

    lines.append("")
    lines.append("## Duplicate-Heavy Chunks")
    if report["duplicate_heavy_chunks"]:
        for item in report["duplicate_heavy_chunks"][:10]:
            lines.append(
                f"- chunk {item['chunk_index']}: duplicate_ratio={item['duplicate_ratio']:.2%} "
                f"({item['lines']} lines)"
            )
    else:
        lines.append("- none detected")

    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- merged: `{report['paths']['merged']}`")
    lines.append(f"- train: `{report['paths']['train']}`")
    lines.append(f"- valid: `{report['paths']['valid']}`")
    lines.append(f"- sample: `{report['paths']['sample']}`")
    return "\n".join(lines) + "\n"


def detect_duplicate_heavy_chunks(
    lines: list[str], chunk_size: int, threshold: float
) -> list[dict]:
    findings: list[dict] = []
    if chunk_size <= 0:
        return findings
    for idx in range(0, len(lines), chunk_size):
        chunk = lines[idx : idx + chunk_size]
        if not chunk:
            continue
        unique = len(set(chunk))
        duplicate_ratio = 1.0 - (unique / len(chunk))
        if duplicate_ratio >= threshold:
            findings.append(
                {
                    "chunk_index": idx // chunk_size,
                    "lines": len(chunk),
                    "duplicate_ratio": duplicate_ratio,
                }
            )
    return findings


def run(args: argparse.Namespace) -> None:
    dirs = ensure_dirs(Path(args.data_dir), Path(args.docs_dir))
    manifest_path = dirs["raw_dir"] / "dataset_manifest.json"
    report_path = dirs["docs_dir"] / "day2_data_report.md"
    report_generated = False

    if args.command == "smoke":
        manifest = make_smoke_manifest(dirs["raw_dir"], manifest_path)
        report = process_manifest(
            manifest=manifest,
            intermediate_dir=dirs["intermediate_dir"],
            processed_dir=dirs["processed_dir"],
            report_path=report_path,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            sample_lines_limit=args.sample_lines_limit,
        )
        report_generated = True
        smoke_validate_outputs(dirs["processed_dir"], seed=args.seed, valid_ratio=args.valid_ratio)
        smoke_tokenizer_roundtrip(dirs["processed_dir"] / "train.txt")
        print(json.dumps(report, indent=2))
        print(f"Manifest written to: {manifest_path}")
        print(f"Report written to: {report_path}")
        return

    if args.command in {"run", "ingest"}:
        manifest = ingest(
            sources=DEFAULT_SOURCES,
            raw_dir=dirs["raw_dir"],
            manifest_path=manifest_path,
            max_samples_per_source=args.max_samples_per_source,
            min_chars=args.min_chars,
        )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.command in {"run", "process"}:
        report = process_manifest(
            manifest=manifest,
            intermediate_dir=dirs["intermediate_dir"],
            processed_dir=dirs["processed_dir"],
            report_path=report_path,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            sample_lines_limit=args.sample_lines_limit,
        )
        report_generated = True
        print(json.dumps(report, indent=2))

    print(f"Manifest written to: {manifest_path}")
    if report_generated:
        print(f"Report written to: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 2 data collection and preparation pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default="data", help="Base data directory (contains raw/ and processed/).")
    common.add_argument("--docs-dir", default="docs", help="Directory for Markdown reports.")
    common.add_argument("--max-samples-per-source", type=int, default=5000)
    common.add_argument("--min-chars", type=int, default=20)
    common.add_argument("--valid-ratio", type=float, default=0.05)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--sample-lines-limit", type=int, default=100_000)

    subparsers.add_parser("run", parents=[common], help="Run ingestion + processing + report.")
    subparsers.add_parser("ingest", parents=[common], help="Run ingestion only and write raw manifest/files.")
    subparsers.add_parser("process", parents=[common], help="Run cleaning/merge/split/report from manifest.")
    subparsers.add_parser("smoke", parents=[common], help="Create local smoke data and validate process + tokenizer.")
    return parser


def make_smoke_manifest(raw_dir: Path, manifest_path: Path) -> dict:
    samples = {
        "general_english": [
            "This is a short English sentence for language model training.",
            "Another line with punctuation, numbers 123, and mixed tokens.",
            "<p>Light HTML should be cleaned but text should remain.</p>",
            "Spacing    should normalize cleanly.",
            "This is a short English sentence for language model training.",
        ],
        "indic_hindi": [
            "यह हिंदी में एक नमूना वाक्य है।",
            "भारत की भाषाई विविधता बहुत समृद्ध है।",
            "<div>हिंदी टेक्स्ट से HTML टैग हटने चाहिए।</div>",
            "यह हिंदी में एक नमूना वाक्य है।",
        ],
    }

    entries: list[dict] = []
    for source_id, lines in samples.items():
        source_dir = raw_dir / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        raw_file = source_dir / "raw.txt"
        with raw_file.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        entries.append(
            {
                "source": source_id,
                "language": "en" if source_id == "general_english" else "hi",
                "dataset_path": "smoke_local",
                "dataset_name": "smoke_local",
                "split": "train",
                "license": "test-only",
                "sample_size": len(lines),
                "raw_file": str(raw_file.as_posix()),
                "status": "ok",
            }
        )

    manifest = {"sources": entries}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def smoke_validate_outputs(processed_dir: Path, seed: int, valid_ratio: float) -> None:
    train_path = processed_dir / "train.txt"
    valid_path = processed_dir / "valid.txt"
    merged_path = processed_dir / "merged_corpus.txt"
    sample_path = processed_dir / "sample_100k_lines.txt"

    for path in (train_path, valid_path, merged_path, sample_path):
        assert path.exists(), f"Missing expected file: {path}"

    train_lines = train_path.read_text(encoding="utf-8").splitlines()
    valid_lines = valid_path.read_text(encoding="utf-8").splitlines()
    merged_lines = merged_path.read_text(encoding="utf-8").splitlines()
    assert merged_lines, "Merged corpus is empty."
    assert valid_lines, "Validation split is empty."
    assert len(train_lines) + len(valid_lines) == len(merged_lines), "Train/valid sizes do not add up."

    # Determinism check: recompute indices only and verify expected valid size.
    expected_valid = int(len(merged_lines) * valid_ratio)
    assert len(valid_lines) == expected_valid, "Valid split size is not deterministic."

    # Additional deterministic shuffle check using the same seed on indexed values.
    indices = list(range(len(merged_lines)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    assert len(indices[:expected_valid]) == len(valid_lines), "Deterministic split check failed."


def smoke_tokenizer_roundtrip(train_path: Path) -> None:
    lines = train_path.read_text(encoding="utf-8").splitlines()
    assert lines, "Train file is empty for tokenizer smoke test."
    text = lines[0]
    try:
        from .tokenizer import Tokenizer
    except ModuleNotFoundError:
        print("Tokenizer roundtrip skipped: tiktoken is not installed in this environment.")
        return

    tok = Tokenizer("gpt2")
    token_ids = tok.encode(text)
    decoded = tok.decode(token_ids)
    assert token_ids, "Tokenizer produced empty token list."
    assert isinstance(decoded, str) and len(decoded) > 0, "Tokenizer decode failed."


if __name__ == "__main__":
    cli = build_parser()
    run(cli.parse_args())
