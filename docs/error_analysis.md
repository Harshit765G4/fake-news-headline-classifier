# Error Analysis — Fake News Headline Classifier

## 1. Objective

This report examines the errors made by TF-IDF + Logistic Regression and DistilBERT on the held-out test set. It also records limitations observed during evaluation on a small external challenge set.

The goal is to understand model behavior, not to use the final test set for model selection or hyperparameter tuning.

## 2. Held-Out Test Results

The test set contains 5,809 headlines.

| Metric | TF-IDF + Logistic Regression | DistilBERT |
|---|---:|---:|
| Accuracy | 94.61% | 98.59% |
| Macro precision | 94.68% | 98.62% |
| Macro recall | 94.49% | 98.55% |
| Macro F1 | 94.57% | 98.58% |
| Misclassified headlines | 313 | 82 |

DistilBERT makes 231 fewer classification errors than the TF-IDF baseline on this test set.

These results measure performance on the held-out dataset. They do not establish equivalent performance on current news, different publishers, or real-world misinformation.

## 3. Error Overlap

| Error category | Number of headlines |
|---|---:|
| Both models correct | 5,471 |
| Both models wrong | 57 |
| TF-IDF wrong, DistilBERT correct | 256 |
| TF-IDF correct, DistilBERT wrong | 25 |

The categories account for all 5,809 test examples.

The 57 shared mistakes are candidates for qualitative review. The 25 examples where TF-IDF succeeds but DistilBERT fails demonstrate that the more accurate model is not better on every individual example.

## 4. Observed Error Patterns

### 4.1 Political and public-affairs headlines

The sampled shared mistakes include headlines about elections, political figures, legislation, international relations, and public policy.

Examples include:

- "Trump luxury hotel opens just blocks from the White House"
- "China Fires Warning At Trump For Contacting Taiwan In Violation Of One-China Policy"
- "Congress Passes Monumental Mental Health Reform Bill"

The DistilBERT-only error sample also includes political and public-affairs headlines.

These examples suggest that political and public-affairs headlines are worth examining further. The samples alone do not establish that this topic is disproportionately error-prone.

### 4.2 Sensational wording and capitalization

Some examples use all-capital phrases, emphatic language, or dramatic framing. These are candidates for further investigation, but capitalization or sensational wording should not by itself be treated as proof that a headline is false.

### 4.3 Shared mistakes

Both models incorrectly classify 57 test examples. Possible explanations include ambiguous wording, dataset labeling limitations, unfamiliar topics, or patterns learned from the same training corpus. These explanations have not yet been confirmed.

### 4.4 DistilBERT-specific mistakes

There are 25 test headlines that TF-IDF classifies correctly but DistilBERT misclassifies. Contextual representations do not guarantee correct predictions on every example.

A representative example from the sampled errors is:

"White House Officials Admit That Trump LIED About Writing His Own Inaugural Address"

The original dataset record and label should be reviewed before assigning a cause to this error.

## 5. External Challenge Evaluation

A separate challenge evaluation contains two manually labeled death-claim headlines. Both are predicted as Real by DistilBERT, despite being labeled Fake in the challenge dataset.

This raises a concern about generalization beyond the held-out test set.

Limitations:

- The challenge contains only two examples.
- It cannot estimate general real-world accuracy.
- Evidence URLs and labels should be independently verified before publication.
- Confidence scores are model outputs, not probabilities that a claim is factually true.

Challenge results must be reported separately from held-out test results.

## 6. Limitations

1. The dataset may contain source-, topic-, or time-specific patterns.
2. Random stratified splitting does not guarantee generalization to unseen publishers or future events.
3. Headline-only classification omits article context, source reliability, corroboration, and external evidence.
4. A model can produce a confident but incorrect prediction.
5. The small external challenge set is insufficient for broad deployment conclusions.
6. A predicted label must not be presented as a verified factual judgment.

## 7. Recommended Future Evaluation

- Build a larger, independently verified external evaluation set.
- Measure performance by topic, source, publication period, and headline length when these attributes are available.
- Investigate shared mistakes and model disagreements using original dataset records.
- Evaluate publisher-held-out or time-based splits.
- Assess probability calibration on a dedicated validation set before using confidence thresholds.
- Consider evidence retrieval and source verification as separate components rather than treating headline classification as fact-checking.

## 8. Conclusion

DistilBERT substantially outperforms the TF-IDF baseline on the held-out test set, but both models make errors. The external challenge examples illustrate that high test accuracy does not guarantee reliable predictions on arbitrary real-world claims.

The system should be presented as an experimental headline classifier, not an automated fact-checker.
