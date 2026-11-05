# Week 2: Designing and Implementing Text Classification Strategies

**Prepared by:** Shaik Tilawath Ahmed  
**Dataset:** Cornell sentence polarity dataset v1.0  
**Task:** Binary sentiment classification  
**Corpus size:** 10,662 snippets (5,331 positive; 5,331 negative)

## Executive summary

This report defines a practical, leakage-aware strategy for classifying short movie-review snippets as positive or negative. It continues the Week 1 EDA work using the same Cornell sentence polarity dataset v1.0. The proposed experiment compares CountVectorizer + Multinomial Naive Bayes, TF-IDF + Multinomial Naive Bayes, TF-IDF + Logistic Regression, and TF-IDF + LinearSVC.

Model selection uses stratified five-fold cross-validation on the training partition, with macro-F1 as the primary selection metric. A separate stratified holdout test set is reserved for one final evaluation after model selection. The companion Python script implements this workflow and exports metrics, predictions, visualizations, and the selected model.

**Execution note:** The full 10,662-record training run was not executed in the authoring environment because external dataset downloads were unavailable. No classifier accuracy or winning model is claimed in this report. The script is designed to generate those values reproducibly when run against the actual corpus.

## 1. Problem statement and objectives

Given a short English-language movie-review snippet, predict whether its sentiment label is positive or negative. This is a supervised binary text-classification problem.

Objectives:
- Establish a reproducible baseline using traditional machine-learning methods.
- Compare count-based and TF-IDF feature representations.
- Evaluate Naive Bayes, Logistic Regression, and LinearSVC.
- Prevent vocabulary/IDF leakage during cross-validation.
- Evaluate aggregate performance and class-specific errors.

## 2. Dataset and experimental design

The Cornell sentence polarity dataset v1.0 contains 5,331 positive and 5,331 negative processed Rotten Tomatoes snippets (10,662 total). Each line is a snippet, generally approximately one sentence. This is the sentence-level v1.0 corpus, not Cornell's separate document-level polarity dataset v2.0, which contains 1,000 positive and 1,000 negative full reviews.

The labels are associated with the overall Fresh/Rotten review rating, so an individual sentence may not always express the same polarity as the full review. This is a known source of label noise.

The Week 1 EDA measured about 18.90 tokens per snippet on average (median 18) and 18,989 token types with its documented tokenizer. The balanced class distribution makes accuracy interpretable, but macro-F1 remains the primary selection metric to weight both classes equally.

## 3. Data preparation and preprocessing

1. Load positive and negative text files separately, infer the label from the source file, and assign stable snippet IDs.
2. Remove empty lines while recording the number skipped. Inspect duplicate and near-duplicate content rather than silently deleting records.
3. Convert text to lowercase and normalize whitespace.
4. Replace URL and email patterns with placeholders if present; remove markup tags while retaining visible text.
5. Remove most non-linguistic symbols but preserve apostrophes, digits, and selected punctuation.
6. Retain negation words such as not, no, never, and contractions because they can reverse sentiment.
7. Tokenize while preserving simple contractions. Stemming and lemmatization are not used in the first baseline.

The script preserves original text for later error analysis and creates a separate normalized-text field.

## 4. Feature extraction

| Method | Representation | Strength | Limitation |
|---|---|---|---|
| Bag of Words / CountVectorizer | Token counts or binary presence | Simple, interpretable baseline | Common words may dominate; no semantics |
| TF-IDF | Term frequency weighted by inverse document frequency | Emphasizes terms distinctive to a snippet | Sparse lexical representation; no synonym/context understanding |
| Word and character n-grams | Unigrams, bigrams, character fragments | Bigrams capture local phrases and negation; character features help spelling variants | Can create many sparse features |
| Pretrained embeddings / transformers | Dense contextual vectors | Can capture semantic similarity and context | More compute and implementation complexity |

**Selected baseline:** TF-IDF with word unigrams and bigrams (ngram_range=(1, 2)), min_df=2, max_df=0.95, sublinear term frequency, and a token pattern that retains contractions. CountVectorizer with the same n-gram range is compared as a count-based alternative. Vectorization is inside each scikit-learn Pipeline so it is fitted independently within every training fold.

## 5. Algorithm selection

| Algorithm | Role | Advantages | Limitations |
|---|---|---|---|
| Multinomial Naive Bayes | Probabilistic sparse-text baseline | Fast, simple, effective with non-negative word features | Conditional-independence assumption |
| Logistic Regression | Linear discriminative model | Regularization, interpretable coefficients, probability output | Requires regularization tuning |
| LinearSVC | Linear maximum-margin classifier | Strong baseline for high-dimensional sparse text | Decision scores are not calibrated probabilities |

A DummyClassifier using the most-frequent class is a floor baseline. Because the corpus is balanced, it should score about 50% accuracy and is not an acceptable predictive model.

**Selection rule:** Rank the candidates by mean stratified cross-validation macro-F1 on the training partition. Use mean accuracy only as a tie-breaker. Do not choose a model by repeatedly examining the final test set.

## 6. Training and validation plan

1. Load and validate the positive and negative files; confirm both classes are present.
2. Normalize text consistently and retain source text and snippet IDs.
3. Create one stratified 80/20 train/test split using random_state=42.
4. Create separate Pipelines for CountVectorizer + MultinomialNB, TF-IDF + MultinomialNB, TF-IDF + LogisticRegression, and TF-IDF + LinearSVC.
5. Run stratified five-fold cross-validation on the 80% training partition.
6. Calculate accuracy, macro precision, macro recall, and macro-F1 for each fold; retain means and standard deviations.
7. Select the best candidate by mean macro-F1 and refit it on all training rows.
8. Predict the untouched 20% test partition once.
9. Report aggregate and per-class metrics and the confusion matrix.
10. Inspect false positives and false negatives for negation, contrast, short text, sarcasm, and likely label noise.
11. Save the fitted pipeline, CV comparison, final metrics, predictions, and model feature weights.

Stratification preserves the class ratio. Five-fold cross-validation provides a more stable view than one validation split. The final test partition remains outside model and hyperparameter selection. If additional tuning is needed, use GridSearchCV with the full vectorizer-classifier Pipeline and training-only validation.

## 7. Evaluation framework

| Metric | Definition | Use |
|---|---|---|
| Accuracy | (TP + TN) / total examples | Overall fraction of correct predictions |
| Precision | TP / (TP + FP) | Reliability of positive predictions |
| Recall | TP / (TP + FN) | Coverage of actual positive examples |
| F1-score | 2 × (Precision × Recall) / (Precision + Recall) | Balances precision and recall |
| Macro-F1 | Unweighted mean of positive and negative class F1 | Primary model-selection metric |
| Weighted-F1 | Class F1 values weighted by support | Support-weighted summary |
| ROC-AUC | Area under ROC using decision scores/probabilities | Ranking quality where score outputs are available |

In the confusion matrix, true positives and true negatives are correct predictions. False positives are negative snippets predicted as positive; false negatives are positive snippets predicted as negative.

## 8. Challenges and mitigation

| Challenge | Why it matters | Mitigation |
|---|---|---|
| Label noise | Sentence sentiment may differ from the overall review rating | Inspect representative errors; describe results as dataset-specific |
| Negation and contrast | “Not good” differs from “good”; contrast can reverse tone | Preserve negators and contractions; test bigrams |
| Short snippets | Limited context makes polarity ambiguous | Report class-wise errors; consider contextual models later |
| Vocabulary sparsity | Rare names and terms create a large feature space | Tune minimum document frequency and regularization |
| Data leakage | Fitting IDF or selecting features before splitting inflates scores | Keep vectorization and classifier in one Pipeline inside CV |
| Domain/time limits | Historical movie-review snippets may not generalize to social posts or current reviews | Validate externally on a newer target-domain dataset |
| Metric imbalance | Accuracy can hide weak performance for one class | Report macro-F1, per-class metrics, and confusion matrix |

## 9. Future enhancements

- Tune vectorizer settings and model regularization using a documented training-only grid search.
- Compare unigram-only, unigram-plus-bigram, binary counts, and character n-gram representations.
- Inspect influential features and review errors for dataset-specific artifacts.
- Evaluate calibrated probabilities and decision thresholds if confidence scores are needed.
- Try word embeddings or a compact transformer only after classical baselines establish a reliable reference.
- Test generalization on a newer movie-review or product-review dataset, checking license and domain differences.

## 10. Reproducibility and implementation outputs

Install the dependencies and run:

~~~bash
pip install -r requirements.txt
python train_classifier.py --download --data_dir data --out_dir outputs
~~~

The script creates:
- cross_validation_comparison.csv: fold means and standard deviations for each candidate.
- test_metrics.json: final holdout metrics, class report, confusion matrix, and run metadata.
- test_predictions.csv: IDs, original text, true labels, predicted labels, and error type.
- confusion_matrix.png and cv_model_comparison.png: visual summaries.
- top_model_features.csv: strongest feature weights when available.
- best_model.joblib: serialized vectorizer-classifier pipeline.
- prepared_dataset.csv: source and normalized text with labels.

A small synthetic fixture was used only to smoke-test the script interface and output generation. Fixture scores are not corpus results and are intentionally excluded from this report. Run the script on the complete Cornell corpus to obtain actual model metrics.

## 11. Conclusion

This strategy defines a practical classical NLP classification experiment for balanced positive/negative movie-review snippets. It compares count and TF-IDF representations, evaluates Naive Bayes, Logistic Regression, and LinearSVC, and uses stratified cross-validation plus an untouched holdout test set. Macro-F1, class-wise results, and confusion-matrix inspection provide a more informative evaluation than accuracy alone. The companion script turns the plan into a reproducible implementation.

## References

1. Cornell University. [Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/).
2. Pang, B. and Lee, L. (2005). *Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales*. Proceedings of ACL.
3. scikit-learn documentation: [TfidfVectorizer and feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html).
4. scikit-learn documentation: [Pipeline and composite estimators](https://scikit-learn.org/stable/modules/compose.html).
