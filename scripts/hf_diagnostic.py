#!/usr/bin/env python
"""Quick diagnostic utility for verifying HuggingFace assets.

This script inspects a HuggingFace dataset repository (defaults to
``omarkamali/wikipedia-monthly``) and reports which artifacts exist for
combinations of dates, languages, and model components. Use it on CI to debug
404 errors before running the full test suite.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Set

from huggingface_hub import list_repo_files
from huggingface_hub.utils import HfHubHTTPError

DEFAULT_REPO_ID = "omarkamali/wikipedia-monthly"
SUPPORTED_COMPONENTS = {"dataset", "markov", "ngram", "tokenizer", "vocabulary"}


@dataclass
class ComponentStatus:
    component: str
    missing: List[str]
    present: List[str]

    @property
    def ok(self) -> bool:
        return not self.missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose HF asset availability")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo to inspect")
    parser.add_argument(
        "--date",
        dest="dates",
        action="append",
        help="Date to inspect (YYYYMMDD). Can be passed multiple times.",
    )
    parser.add_argument(
        "--lang",
        dest="langs",
        action="append",
        help="Language code to inspect. Can be passed multiple times.",
    )
    parser.add_argument(
        "--component",
        dest="components",
        action="append",
        choices=sorted(SUPPORTED_COMPONENTS),
        help="Component(s) to inspect (defaults to all).",
    )
    parser.add_argument(
        "--depth",
        dest="depths",
        action="append",
        type=int,
        help="Markov depths to check (defaults to 1-5 if omitted).",
    )
    parser.add_argument(
        "--gram",
        dest="grams",
        action="append",
        type=int,
        help="N-gram sizes to check (defaults to 2-4 if omitted).",
    )
    parser.add_argument(
        "--vocab-size",
        dest="vocab_sizes",
        action="append",
        type=int,
        help="Tokenizer vocab sizes to check (defaults to 8k/16k/32k if omitted).",
    )
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        help="Variant folders to inspect (defaults to ['word', 'subword']).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress OK statuses, only print failures.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="List present files for each component (verbose).",
    )
    return parser.parse_args()


def ensure_defaults(args: argparse.Namespace) -> None:
    if not args.dates:
        args.dates = ["20251201"]
    if not args.langs:
        args.langs = ["ary"]
    if not args.components:
        args.components = sorted(SUPPORTED_COMPONENTS)
    if not args.depths:
        args.depths = [1, 2, 3, 4, 5]
    if not args.grams:
        args.grams = [2, 3, 4]
    if not args.vocab_sizes:
        args.vocab_sizes = [8000, 16000, 32000]
    if not args.variants:
        args.variants = ["word", "subword"]


def build_file_set(repo_id: str) -> Set[str]:
    try:
        repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    except HfHubHTTPError as err:
        raise SystemExit(f"Failed to list files for {repo_id}: {err}")
    return set(repo_files)


def check_dataset(date: str, lang: str, files: Set[str]) -> ComponentStatus:
    sizes = {"1k": "1000k.parquet", "5k": "5000k.parquet", "10k": "10000k.parquet"}
    paths = [f"{date}/{lang}/dataset/{fname}" for fname in sizes.values()]
    return partition_status("dataset", paths, files)


def check_markov(date: str, lang: str, files: Set[str], depths: Iterable[int], variants: Iterable[str]) -> ComponentStatus:
    required = []
    for depth in depths:
        for folder in ["markov", *[f"{variant}_markov" for variant in variants]]:
            required.append(f"{date}/{lang}/models/{folder}/{lang}_markov{depth}_transitions.parquet")
            required.append(f"{date}/{lang}/models/{folder}/{lang}_markov{depth}_metadata.json")
    return partition_status("markov", required, files)


def check_ngram(date: str, lang: str, files: Set[str], grams: Iterable[int], variants: Iterable[str]) -> ComponentStatus:
    required = []
    for gram in grams:
        for folder in ["ngram", *[f"{variant}_ngram" for variant in variants]]:
            required.append(f"{date}/{lang}/models/{folder}/{lang}_{gram}gram_model.parquet")
            required.append(f"{date}/{lang}/models/{folder}/{lang}_{gram}gram_metadata.json")
    return partition_status("ngram", required, files)


def check_tokenizer(date: str, lang: str, files: Set[str], vocab_sizes: Iterable[int]) -> ComponentStatus:
    required = []
    for vocab_size in vocab_sizes:
        suffix = f"{vocab_size // 1000}k"
        required.append(f"{date}/{lang}/models/tokenizer/{lang}_tokenizer_{suffix}.model")
        required.append(f"{date}/{lang}/models/tokenizer/{lang}_tokenizer_{suffix}.vocab")
    return partition_status("tokenizer", required, files)


def check_vocabulary(date: str, lang: str, files: Set[str]) -> ComponentStatus:
    required = [
        f"{date}/{lang}/models/vocabulary/{lang}_vocabulary.json",
        f"{date}/{lang}/models/vocabulary/{lang}_vocabulary_metadata.json",
    ]
    return partition_status("vocabulary", required, files)


def partition_status(name: str, paths: Iterable[str], files: Set[str]) -> ComponentStatus:
    missing, present = [], []
    for path in paths:
        (present if path in files else missing).append(path)
    return ComponentStatus(name, missing, present)


def describe_status(date: str, lang: str, status: ComponentStatus, quiet: bool, show_files: bool) -> None:
    prefix = f"[{date}][{lang}][{status.component}]"
    if status.ok:
        if not quiet:
            print(f"{prefix} OK ({len(status.present)} files)")
            if show_files:
                for path in status.present:
                    print(f"    ✅ {path}")
    else:
        print(f"{prefix} MISSING {len(status.missing)} files")
        for path in status.missing:
            print(f"    ❌ {path}")
        if show_files and status.present:
            print(f"{prefix} present files:")
            for path in status.present:
                print(f"    ✅ {path}")


def main() -> None:
    args = parse_args()
    ensure_defaults(args)
    files = build_file_set(args.repo_id)

    checkers = {
        "dataset": lambda d, l: check_dataset(d, l, files),
        "markov": lambda d, l: check_markov(d, l, files, args.depths, args.variants),
        "ngram": lambda d, l: check_ngram(d, l, files, args.grams, args.variants),
        "tokenizer": lambda d, l: check_tokenizer(d, l, files, args.vocab_sizes),
        "vocabulary": lambda d, l: check_vocabulary(d, l, files),
    }

    summary_counts = defaultdict(int)

    for date in args.dates:
        for lang in args.langs:
            for component in args.components:
                status = checkers[component](date, lang)
                describe_status(date, lang, status, args.quiet, args.show_files)
                summary_counts[(component, status.ok)] += 1

    print("\nSummary:")
    for component in args.components:
        ok = summary_counts.get((component, True), 0)
        bad = summary_counts.get((component, False), 0)
        print(f"  {component}: {ok} OK / {bad} missing")


if __name__ == "__main__":
    main()
