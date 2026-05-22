#!/usr/bin/env python3
# You can pass in the file path and model name
# python scripts/generate_dataset_variants.py envs/MATH/dataset/test_aime.jsonl --models "Qwen3-0.6B,Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-14B,Qwen3-32B"

# Example command to remove generated files:
# rm ./envs/MATH/dataset/aime_Qwen3*.jsonl
"""Duplicate a JSONL dataset with different LM/beam configurations."""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict

DEFAULT_MODELS = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
]

BEAM_VARIANTS: List[Dict[str, int]] = [
    {"QP": 1, "CP": 64, "BS": 4},
    #{"QP": 2, "CP": 32, "BS": 4},
    #{"QP": 4, "CP": 16, "BS": 4},
    #{"QP": 8, "CP": 8, "BS": 2},
    #{"QP": 16, "CP": 4, "BS": 2},
    #{"QP": 32, "CP": 2, "BS": 1},
    #{"QP": 64, "CP": 1, "BS": 1},
    #{"QP": 16, "CP": 1, "BS": 1},
    #{"QP": 1, "CP": 16, "BS": 1},
    #{"QP": 2, "CP": 8, "BS": 2},
    #{"QP": 4, "CP": 4, "BS": 2},
    #{"QP": 8, "CP": 2, "BS": 1},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source JSONL file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (default: same folder as input)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix (default: input stem with leading 'test_' removed)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated LM names (default: %(default)s)",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
    return data


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def sanitize_prefix(stem: str) -> str:
    if stem.startswith("test_"):
        return stem[5:]
    return stem


def main() -> None:
    args = parse_args()
    source_path: Path = args.input
    if not source_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {source_path}")

    data = read_jsonl(source_path)

    output_dir = args.output_dir or source_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or sanitize_prefix(source_path.stem)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    total_written = 0
    for model in models:
        for beam_cfg in BEAM_VARIANTS:
            updated_records: List[Dict] = []
            for record in data:
                new_record = dict(record)
                new_record["lm"] = model
                new_record["beam"] = dict(beam_cfg)
                updated_records.append(new_record)

            file_name = (
                f"{prefix}_{model}_QP{beam_cfg['QP']}_CP{beam_cfg['CP']}_BS{beam_cfg['BS']}.jsonl"
            )
            write_jsonl(output_dir / file_name, updated_records)
            total_written += 1

    print(f"Generated {total_written} files in {output_dir}")


if __name__ == "__main__":
    main()
