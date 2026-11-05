# Week 2 — Text Classification Strategy

This folder extends the Week 1 NLP EDA project with a reproducible supervised sentiment-classification strategy and an executable training/evaluation script.

## Deliverables

- **Word report:** Week2_Text_Classification_Strategy_Report.docx is supplied separately for upload to the internship portal.
- **Full readable report:** [REPORT.md](REPORT.md)
- **Runnable implementation:** [train_classifier.py](train_classifier.py)
- **Submission description (more than 200 words):** [submission_description.txt](submission_description.txt)

## Dataset

Uses the same Cornell **sentence polarity dataset v1.0** as Week 1: 5,331 positive and 5,331 negative Rotten Tomatoes snippets (10,662 total). This is the sentence-level v1.0 corpus, not the separate 2,000 full-review v2.0 corpus.

Primary source: https://www.cs.cornell.edu/people/pabo/movie-review-data/

The script tries Cornell's archive first and falls back to public plain-text files hosted in the Elijas sentence-polarity dataset repository.

## Run

Python 3.9 or newer:

~~~bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_classifier.py --download --data_dir data --out_dir outputs
~~~

To use already-downloaded files, place rt-polarity.pos and rt-polarity.neg in data/ (or the mirror filenames) and omit --download.

## Implemented experiment

The script compares a dummy baseline plus:
1. Most-frequent-class DummyClassifier baseline
2. CountVectorizer + Multinomial Naive Bayes
3. TF-IDF + Multinomial Naive Bayes
4. TF-IDF + Logistic Regression
5. TF-IDF + LinearSVC

It uses a stratified 80/20 train/test split, stratified 5-fold cross-validation on the training partition, macro-F1 for model selection, and an untouched final test partition. Vectorization is inside each scikit-learn Pipeline to avoid fitting vocabulary/IDF statistics on validation or test data.

## Output files

The script creates:
- cross_validation_comparison.csv
- test_metrics.json
- test_predictions.csv
- confusion_matrix.png
- cv_model_comparison.png
- top_model_features.csv
- best_model.joblib
- prepared_dataset.csv

No full-corpus classifier performance is claimed in the report until this script is run against the full public dataset. The script was smoke-tested with synthetic rows only to verify its execution path and artifact generation; those fixture scores are not corpus results.

## Attribution

Pang, B. and Lee, L. (2005). *Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales*. Proceedings of ACL. Dataset details are on the Cornell Movie Review Data page.
