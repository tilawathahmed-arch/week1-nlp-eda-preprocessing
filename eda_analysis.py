#!/usr/bin/env python3
"""Reproducible EDA and preprocessing for the Cornell sentence-polarity dataset v1.0.

Input options:
  * rt-polarity.pos and rt-polarity.neg (official archive filenames), or
  * rt-polarity-pos.txt and rt-polarity-neg.txt (public mirror filenames).

With --download, the script tries Cornell first and then a public text mirror.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CORNELL_ARCHIVE = (
    "https://www.cs.cornell.edu/people/pabo/movie-review-data/"
    "rt-polaritydata.tar.gz"
)
MIRROR_BASE = (
    "https://raw.githubusercontent.com/abromberg/sentiment_analysis_python/"
    "master/polarityData/rt-polaritydata/"
)
MIRROR_FILES = {
    "pos": "rt-polarity-pos.txt",
    "neg": "rt-polarity-neg.txt",
}

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
TAG_RE = re.compile(r"<[^>]+>")

# Negators are intentionally not part of this list.
STOP_WORDS = set("""a about above after again against all am an and any are as at be
because been before being below between both but by can could did do does doing down
during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself of off on
once only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves""".split())


def locate_file(data_dir: Path, label: str) -> Path | None:
    names = {
        "pos": ["rt-polarity.pos", "rt-polarity-pos.txt", "rt-polarity.pos.txt"],
        "neg": ["rt-polarity.neg", "rt-polarity-neg.txt", "rt-polarity.neg.txt"],
    }[label]
    roots = [data_dir] + [p for p in data_dir.rglob("*") if p.is_dir()]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract regular files/directories only, preventing path traversal and links."""
    base = destination.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            continue
        target = (destination / member.name).resolve()
        if target == base or base in target.parents:
            tar.extract(member, destination)


def download_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "rt-polaritydata.tar.gz"
    try:
        print(f"Downloading official Cornell archive: {CORNELL_ARCHIVE}")
        urllib.request.urlretrieve(CORNELL_ARCHIVE, archive)
        with tarfile.open(archive, "r:gz") as tar:
            safe_extract(tar, data_dir)
        if locate_file(data_dir, "pos") and locate_file(data_dir, "neg"):
            print("Official Cornell archive downloaded and extracted.")
            return
        raise FileNotFoundError("Expected positive/negative files were not found.")
    except Exception as exc:
        print(f"Official archive download failed ({exc}). Trying public mirror...")
        for label, filename in MIRROR_FILES.items():
            target = data_dir / filename
            urllib.request.urlretrieve(MIRROR_BASE + filename, target)
            print(f"Downloaded {label} snippets to {target}")


def load_snippets(data_dir: Path) -> pd.DataFrame:
    rows = []
    for label in ("neg", "pos"):
        path = locate_file(data_dir, label)
        if path is None:
            raise FileNotFoundError(
                f"Missing {label} file in {data_dir}. Run with --download or "
                "place the positive/negative source files in that folder."
            )
        content = path.read_text(encoding="latin-1", errors="replace")
        for index, line in enumerate(content.splitlines(), start=1):
            rows.append({
                "snippet_id": f"{label}_{index:05d}",
                "label": "negative" if label == "neg" else "positive",
                "source_file": path.name,
                "raw_text": line.strip(),
            })
    return pd.DataFrame(rows)


def clean_text(value: str) -> str:
    """Normalize formatting conservatively; preserve contractions and negation."""
    text = (value or "").lower()
    text = URL_RE.sub(" URLTOKEN ", text)
    text = EMAIL_RE.sub(" EMAILTOKEN ", text)
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9'!?.,;:\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def describe(values: list[int]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": int(arr.min()) if arr.size else 0,
        "q1": round(float(np.percentile(arr, 25)), 2) if arr.size else 0,
        "median": round(float(np.percentile(arr, 50)), 2) if arr.size else 0,
        "mean": round(float(arr.mean()), 2) if arr.size else 0,
        "q3": round(float(np.percentile(arr, 75)), 2) if arr.size else 0,
        "max": int(arr.max()) if arr.size else 0,
    }


def count_ngrams(
    token_rows: list[list[str]], n: int, remove_stopwords: bool = False
) -> Counter:
    counts: Counter = Counter()
    for tokens in token_rows:
        for index in range(len(tokens) - n + 1):
            window = tokens[index:index + n]
            if remove_stopwords and any(token in STOP_WORDS for token in window):
                continue
            counts[" ".join(window)] += 1
    return counts


def write_counter(counter: Counter, path: Path) -> None:
    pd.DataFrame(counter.most_common(), columns=["term", "count"]).to_csv(
        path, index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EDA on Cornell sentence polarity dataset v1.0"
    )
    parser.add_argument(
        "--data_dir", default="data",
        help="Folder containing the positive and negative source files"
    )
    parser.add_argument(
        "--out_dir", default="outputs",
        help="Folder for cleaned data, tables, charts, and JSON"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download the Cornell archive, falling back to the public mirror"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download and (
        locate_file(data_dir, "pos") is None or locate_file(data_dir, "neg") is None
    ):
        download_dataset(data_dir)

    df = load_snippets(data_dir)
    df["is_blank"] = df["raw_text"].fillna("").str.strip().eq("")
    df["has_markup"] = df["raw_text"].fillna("").map(
        lambda value: bool(TAG_RE.search(value))
    )
    df["has_url"] = df["raw_text"].fillna("").map(
        lambda value: bool(URL_RE.search(value))
    )
    df["has_email"] = df["raw_text"].fillna("").map(
        lambda value: bool(EMAIL_RE.search(value))
    )
    df["has_digit"] = df["raw_text"].fillna("").map(
        lambda value: bool(re.search(r"\d", value))
    )
    df["clean_text"] = df["raw_text"].fillna("").map(clean_text)
    df["tokens"] = df["clean_text"].map(tokenize)
    df["char_count"] = df["raw_text"].fillna("").str.len()
    df["clean_char_count"] = df["clean_text"].str.len()
    df["token_count"] = df["tokens"].map(len)
    df["unique_token_count"] = df["tokens"].map(
        lambda tokens: len(set(tokens))
    )
    df["negation_count"] = df["clean_text"].map(
        lambda value: len(re.findall(r"\b(?:not|no|nor|never|n't)\b", value))
    )
    df["exclamation_count"] = df["raw_text"].map(
        lambda value: (value or "").count("!")
    )
    df["question_count"] = df["raw_text"].map(
        lambda value: (value or "").count("?")
    )
    df["duplicate_raw"] = df["raw_text"].duplicated(keep=False)
    df["duplicate_clean"] = df["clean_text"].duplicated(keep=False)

    token_rows = df["tokens"].tolist()
    token_flat = [token for row in token_rows for token in row]
    unigram_all = Counter(token_flat)
    bigram_all = count_ngrams(token_rows, 2)
    unigram_content = Counter(
        token for token in token_flat if token not in STOP_WORDS
    )
    bigram_content = count_ngrams(token_rows, 2, remove_stopwords=True)

    by_class = {}
    for label, group in df.groupby("label", sort=True):
        by_class[label] = {
            "documents": int(len(group)),
            "characters": describe(group["char_count"].tolist()),
            "tokens": describe(group["token_count"].tolist()),
            "mean_unique_tokens": round(
                float(group["unique_token_count"].mean()), 2
            ),
            "mean_negations": round(float(group["negation_count"].mean()), 4),
            "mean_exclamations": round(
                float(group["exclamation_count"].mean()), 4
            ),
            "mean_questions": round(float(group["question_count"].mean()), 4),
        }

    summary = {
        "dataset": "Cornell sentence polarity dataset v1.0 (Rotten Tomatoes snippets)",
        "dataset_source": "https://www.cs.cornell.edu/people/pabo/movie-review-data/",
        "documents": int(len(df)),
        "class_counts": {
            str(key): int(value) for key, value in df["label"].value_counts().items()
        },
        "class_percentages": {
            str(key): round(float(value / len(df) * 100), 2)
            for key, value in df["label"].value_counts().items()
        },
        "missing_or_blank": int(df["is_blank"].sum()),
        "duplicate_raw_rows": int(df["duplicate_raw"].sum()),
        "duplicate_clean_rows": int(df["duplicate_clean"].sum()),
        "character_length": describe(df["char_count"].tolist()),
        "token_length": describe(df["token_count"].tolist()),
        "mean_unique_tokens_per_snippet": round(
            float(df["unique_token_count"].mean()), 2
        ),
        "total_tokens": int(len(token_flat)),
        "vocabulary_size": int(len(set(token_flat))),
        "quality_flags": {
            "rows_with_markup": int(df["has_markup"].sum()),
            "rows_with_urls": int(df["has_url"].sum()),
            "rows_with_email_like_strings": int(df["has_email"].sum()),
            "rows_with_digits": int(df["has_digit"].sum()),
            "total_exclamation_marks": int(df["exclamation_count"].sum()),
            "total_question_marks": int(df["question_count"].sum()),
            "total_negation_markers": int(df["negation_count"].sum()),
        },
        "by_class": by_class,
        "top_unigrams_raw": [
            {"term": term, "count": int(count)}
            for term, count in unigram_all.most_common(20)
        ],
        "top_unigrams_content": [
            {"term": term, "count": int(count)}
            for term, count in unigram_content.most_common(20)
        ],
        "top_bigrams_raw": [
            {"term": term, "count": int(count)}
            for term, count in bigram_all.most_common(20)
        ],
        "top_bigrams_content": [
            {"term": term, "count": int(count)}
            for term, count in bigram_content.most_common(20)
        ],
        "run_metadata": {
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "matplotlib_version": plt.matplotlib.__version__,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    export = df.drop(columns=["tokens"])
    export.to_csv(
        out_dir / "cleaned_snippets.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL
    )
    write_counter(unigram_all, out_dir / "unigram_frequencies_raw.csv")
    write_counter(bigram_all, out_dir / "bigram_frequencies_raw.csv")
    write_counter(unigram_content, out_dir / "unigram_frequencies_content.csv")
    write_counter(bigram_content, out_dir / "bigram_frequencies_content.csv")

    class_features = (
        df.groupby("label")[
            [
                "token_count", "unique_token_count", "negation_count",
                "exclamation_count", "question_count"
            ]
        ].mean().round(3)
    )
    class_features.to_csv(out_dir / "class_feature_means.csv")

    quality = pd.DataFrame(
        [
            ("Blank snippets", int(df["is_blank"].sum())),
            ("Rows with markup", int(df["has_markup"].sum())),
            ("Rows with URLs", int(df["has_url"].sum())),
            ("Rows with email-like strings", int(df["has_email"].sum())),
            ("Rows containing digits", int(df["has_digit"].sum())),
            ("Duplicate raw-text rows", int(df["duplicate_raw"].sum())),
            ("Duplicate cleaned-text rows", int(df["duplicate_clean"].sum())),
        ],
        columns=["check", "count"],
    )
    quality.to_csv(out_dir / "data_quality_checks.csv", index=False)

    # Figure 1: class distribution.
    class_counts = df["label"].value_counts().reindex(["negative", "positive"])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(
        class_counts.index.str.title(), class_counts.values,
        color=["#D66A63", "#2C8C84"]
    )
    ax.set_title("Sentiment label distribution")
    ax.set_ylabel("Number of snippets")
    label_offset = max(1.0, float(max(class_counts.values)) * 0.02)
    ax.set_ylim(0, float(max(class_counts.values)) * 1.14 + label_offset)
    ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, class_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + label_offset,
            f"{value:,}", ha="center", fontsize=10
        )
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_class_distribution.png", dpi=180)
    plt.close(fig)

    # Figure 2: token-length histogram by class.
    edges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]
    labels = [
        "0-4", "5-9", "10-14", "15-19", "20-24", "25-29",
        "30-34", "35-39", "40-44", "45-49", "50-59"
    ]
    hist_rows = []
    for label in ("negative", "positive"):
        values = df.loc[df.label == label, "token_count"].to_numpy()
        counts = [
            int(((values >= low) & (values < high)).sum())
            for low, high in zip(edges[:-1], edges[1:])
        ]
        hist_rows.append(counts)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(labels))
    width = .38
    ax.bar(x - width / 2, hist_rows[0], width, label="Negative", color="#D66A63")
    ax.bar(x + width / 2, hist_rows[1], width, label="Positive", color="#2C8C84")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Tokens per snippet")
    ax.set_ylabel("Number of snippets")
    ax.set_title("Snippet token-length distribution by sentiment")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_token_length_distribution.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(
        {"range": labels, "negative": hist_rows[0], "positive": hist_rows[1]}
    ).to_csv(out_dir / "token_length_histogram.csv", index=False)

    # Figure 3: content unigrams. Negators are preserved.
    top_uni = unigram_content.most_common(15)
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    terms, values = zip(*reversed(top_uni))
    ax.barh(terms, values, color="#2C8C84")
    ax.set_title("Most frequent content unigrams")
    ax.set_xlabel("Frequency")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_top_unigrams.png", dpi=180)
    plt.close(fig)

    # Figure 4: adjacent content bigrams. Pairs with stopwords are omitted.
    top_bi = bigram_content.most_common(15)
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    terms, values = zip(*reversed(top_bi))
    ax.barh(terms, values, color="#4D78A8")
    ax.set_title("Most frequent content bigrams")
    ax.set_xlabel("Frequency")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_top_bigrams.png", dpi=180)
    plt.close(fig)

    # Figure 5: simple language cues by class.
    cue_frame = (
        df.groupby("label")[
            ["negation_count", "question_count", "exclamation_count"]
        ].mean().reindex(["negative", "positive"])
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(2)
    width = .24
    for index, (column, title, color) in enumerate(
        [
            ("negation_count", "Negation markers", "#D66A63"),
            ("question_count", "Question marks", "#4D78A8"),
            ("exclamation_count", "Exclamation marks", "#2C8C84"),
        ]
    ):
        ax.bar(
            x + (index - 1) * width, cue_frame[column].values,
            width, label=title, color=color
        )
    ax.set_xticks(x, ["Negative", "Positive"])
    ax.set_ylabel("Average count per snippet")
    ax.set_title("Exploratory language cues by sentiment")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_language_cues.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nOutputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
