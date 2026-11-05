# Week 1: NLP Exploratory Data Analysis & Preprocessing

This project performs reproducible exploratory data analysis (EDA) and preprocessing on the **Cornell sentence polarity dataset v1.0** (Pang & Lee; Rotten Tomatoes snippets).

> This is the **sentence-level v1.0 corpus**, not Cornell's separate document-level movie-review polarity dataset v2.0.

## Dataset

- **10,662 snippets**: 5,331 positive and 5,331 negative.
- One snippet per line, generally approximately one sentence.
- Labels are derived from Rotten Tomatoes Fresh/Rotten review ratings.
- Official source: [Cornell Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/).
- Dataset documentation: [official v1.0 README](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.README.1.0.txt).
- Retrieval mirror used to validate the measured snapshot: [public plain-text files](https://github.com/abromberg/sentiment_analysis_python/tree/master/polarityData/rt-polaritydata/).

The script first attempts the official Cornell archive. When that endpoint is unavailable, it downloads the two plain-text class files from the public mirror. The mirror is a retrieval fallback; Cornell is the dataset's primary source.

## Project files

```text
.
├── README.md
├── eda_analysis.py
├── requirements.txt
├── submission_description.txt
└── outputs/
    ├── summary.json
    ├── class_distribution.csv
    ├── class_feature_means.csv
    ├── data_quality_checks.csv
    ├── token_length_histogram.csv
    ├── top_content_unigrams.csv
    └── top_content_bigrams.csv
```

The full Word report is provided separately as **Week1_NLP_EDA_Report.docx**. The script recreates the charts and data exports.

## Run locally

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python eda_analysis.py --download --data_dir ./data --out_dir ./outputs
```

You can also manually place the official `rt-polarity.pos` and `rt-polarity.neg` files (or the mirror's `rt-polarity-pos.txt` and `rt-polarity-neg.txt`) in the data folder and run without `--download`.

## What the analysis produces

- Dataset shape, label balance, and per-class descriptive statistics.
- Character/token length distributions and lexical diversity.
- Raw and content-word unigram and adjacent-bigram frequencies.
- Data-quality checks for blank rows, duplicates, markup, URLs, email-like strings, and digit-bearing rows.
- Negation, question-mark, and exclamation-mark summaries by class.
- A cleaned CSV that retains original text, normalized text, labels, IDs, and quality flags.
- JSON metrics, CSV tables, and five PNG charts.

## Measured results

The included snapshot was calculated from both complete source files (5,331 lines per class) using the documented tokenizer and normalization rules.

| Measure | Result |
|---|---:|
| Snippets | 10,662 |
| Negative / positive | 5,331 / 5,331 (50% each) |
| Total tokens | 201,517 |
| Vocabulary size | 18,989 |
| Mean / median tokens | 18.90 / 18 |
| Mean / median characters | 114.16 / 111 |
| Blank rows | 0 |
| Exact duplicate rows (raw / normalized) | 0 / 0 |
| Rows with markup-like tags | 4 |
| Rows with URL / email-like strings | 0 / 0 |
| Rows containing digits | 559 |

The most frequent content unigrams include **film** (1,462), **movie** (1,327), **it's** (1,134), **not** (824), and **story** (481). Leading content bigrams include **it's not** (75), **romantic comedy** (64), and **writer director** (49). Negation is deliberately retained because it can reverse sentiment.

These are descriptive corpus statistics, not classifier-performance metrics. The dataset is balanced by design and its labels may be noisy because sentence snippets inherit the overall review's Fresh/Rotten label.

## Modeling recommendations

A sensible next experiment is TF-IDF with unigram/bigram features and logistic regression. Use stratified splits, fit the vectorizer only on training folds, report class-wise precision/recall/F1 and a confusion matrix, and inspect errors involving negation, sarcasm, contrast, and context.

## Attribution

Pang, B. and Lee, L. (2005). *Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales*. Proceedings of ACL. Consult the official Cornell README for dataset-specific details and attribution.
