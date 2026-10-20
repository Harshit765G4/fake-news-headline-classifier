# Fake News Headline Classifier

An end-to-end NLP project comparing **TF-IDF + Logistic Regression** with **fine-tuned DistilBERT** to classify news headlines as `Fake` or `Real`. The project includes data auditing, preprocessing, exploratory analysis, model training and evaluation, error analysis, LIME explainability, automated tests, and an interactive Streamlit application.

> [!WARNING]
> **This is an educational text-classification project, not a fact-checking system.** A headline alone cannot establish whether a claim is true. Model confidence is not the probability that a claim is factual. Verify important claims using credible reporting and primary sources.

## Contents

- [Highlights](#highlights)
- [Results](#results)
- [Application features](#application-features)
- [Dataset and methodology](#dataset-and-methodology)
- [Project architecture](#project-architecture)
- [Getting started](#getting-started)
- [Reproduce the pipeline](#reproduce-the-pipeline)
- [Run the application](#run-the-application)
- [Run tests](#run-tests)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Future work](#future-work)
- [Repository and license](#repository-and-license)

## Highlights

- Compare a traditional sparse-text baseline against a transformer model.
- Audit missing values, duplicate headlines, normalized duplicates, and conflicting labels.
- Create stratified train, validation, and test splits and check for normalized-headline overlap.
- Evaluate with accuracy, macro precision, macro recall, macro F1, and confusion matrices.
- Inspect test-set errors and compare where the two models succeed or fail.
- Explain TF-IDF baseline predictions with LIME.
- Classify a single headline or a CSV batch through Streamlit.
- Review evaluation metrics and confusion matrices in an interactive dashboard.
- Validate prediction and CSV-handling behavior with pytest.

## Results

Both models were evaluated on the same held-out test set of **5,809 headlines**.

| Metric | TF-IDF + Logistic Regression | DistilBERT |
|---|---:|---:|
| Accuracy | 94.61% | **98.59%** |
| Macro precision | 94.68% | **98.62%** |
| Macro recall | 94.49% | **98.55%** |
| Macro F1-score | 94.57% | **98.58%** |
| Misclassified test examples | 313 | **82** |

On this test split, DistilBERT made 231 fewer errors than the TF-IDF baseline. These figures describe performance on this dataset and split only; they do **not** measure real-world misinformation detection accuracy.

### Confusion matrices

**TF-IDF + Logistic Regression**

![TF-IDF baseline test confusion matrix](reports/figures/tfidf_baseline_test_confusion_matrix.png)

**DistilBERT**

![DistilBERT test confusion matrix](reports/figures/distilbert_test_confusion_matrix.png)

### Error analysis

The saved test-set analysis reported the following comparison:

| Error category | Count |
|---|---:|
| Both models correct | 5,471 |
| Both models wrong | 57 |
| TF-IDF wrong, DistilBERT correct | 256 |
| TF-IDF correct, DistilBERT wrong | 25 |

A separate, very small external challenge set exposed incorrect, highly confident DistilBERT predictions for two death-claim headlines. Two examples are not enough to estimate generalization, but they demonstrate why confidence and held-out test scores must not be mistaken for factual verification.

See [`docs/error_analysis.md`](docs/error_analysis.md) for the qualitative analysis, where available in the checkout.

## Application features

The Streamlit app is launched from `app/app.py` and provides:

- **Single headline prediction:** predicted class and model scores.
- **Batch CSV prediction:** upload a CSV with a `headline` column and download a prediction file; uploads are limited to 5,000 rows.
- **LIME explanation:** inspect word-level contributions for the TF-IDF baseline. This explanation does not describe DistilBERT.
- **Evaluation dashboard:** compare metrics and view the saved confusion matrices.
- **Input validation:** report missing columns, invalid CSV files, and blank headline rows.

Blank headlines are not meaningful examples and should not be interpreted as valid predictions. Probability scores can be overconfident, especially on headlines that differ from the training distribution.

## Dataset and methodology

### Dataset

This project uses the **ISOT Fake News Dataset**, with separate fake and real article collections. The pipeline works with these local source files:

```text
data/raw/Fake.csv
data/raw/True.csv
```

The raw dataset is not included in this GitHub repository. Obtain it from an authorized source and comply with its license and usage terms.

### Preprocessing and split

The processed headline dataset contains **38,725 records** after headline normalization, duplicate handling, and conflict handling.

| Split | Total | Fake | Real |
|---|---:|---:|---:|
| Training | 27,107 | 12,531 | 14,576 |
| Validation | 5,809 | 2,685 | 3,124 |
| Test | 5,809 | 2,685 | 3,124 |
| **Total** | **38,725** | **17,901** | **20,824** |

The pipeline normalizes headline text, audits duplicates, handles conflicting labels, removes unusable records, and creates stratified splits. A normalized-headline overlap check was performed across the splits.

Deduplicating by normalized headline can remove distinct articles that happen to use the same headline. A random split also does not guarantee that publishers, topics, events, or time periods are independent across train and test sets.

### Model 1: TF-IDF + Logistic Regression

The baseline uses a scikit-learn pipeline with:

- TF-IDF unigrams and bigrams.
- Maximum vocabulary size of 100,000 features.
- Minimum document frequency of 2.
- Sublinear term-frequency scaling.
- Logistic Regression with `max_iter=1000`.

This is a lightweight, interpretable point of comparison for the transformer model.

### Model 2: DistilBERT

The transformer model is based on `distilbert-base-uncased` with a binary classification head:

| Setting | Value |
|---|---|
| Maximum sequence length | 64 tokens |
| Training batch size | 8 |
| Epochs | 2 |
| Learning rate | `2e-5` |
| Checkpoint selection | Validation performance |

The final reported metrics were calculated on the held-out test set. Avoid using that test set to choose new hyperparameters; use a validation set for model development and reserve a genuinely untouched dataset for further final evaluation.

## Project architecture

```text
fake-news-headline-classifier/
├── app/
│   └── app.py
├── data/
│   ├── raw/                 # Local source dataset; not committed
│   ├── interim/             # Generated intermediate data; not committed
│   └── processed/           # Generated datasets; not committed
├── docs/
│   └── error_analysis.md
├── models/                  # Locally generated model artifacts; not committed
├── reports/
│   └── figures/             # Includes confusion matrices shown above
├── src/
│   └── fake_news/
│       ├── analyze_errors.py
│       ├── batch_predict.py
│       ├── data_audit.py
│       ├── diagnose_predictions.py
│       ├── evaluate_challenge.py
│       ├── evaluate_test.py
│       ├── explain_lime.py
│       ├── explore_dataset.py
│       ├── predict.py
│       ├── prepare_dataset.py
│       ├── split_dataset.py
│       └── train_baseline.py
│       └── train_transformer.py
├── tests/
│   ├── gpu_diagnostic.py
│   ├── test_batch_predict.py
│   └── test_predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

Some reports, local datasets, and model checkpoints are generated during development and intentionally excluded from version control. The exact set of generated files present locally may differ from a fresh clone.

## Getting started

### Requirements

- Python 3.11 recommended.
- Git.
- Internet access for installing dependencies and downloading pretrained model files.
- Enough RAM and disk space for the dataset and model artifacts.
- An NVIDIA GPU is optional; CPU training and inference can be substantially slower.

### 1. Clone the repository

```powershell
git clone https://github.com/Harshit765G4/fake-news-headline-classifier.git
cd fake-news-headline-classifier
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, consult Microsoft's execution-policy guidance or activate the environment using a terminal appropriate for your system. Do not weaken system-wide security settings just to activate a virtual environment.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

The project also uses **PyTorch** for DistilBERT. Install a PyTorch build suitable for your operating system and CPU/GPU configuration by following the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). Verify the installation:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The first transformer run may download pretrained weights from Hugging Face and therefore requires network access.

## Reproduce the pipeline

The commands below describe the intended workflow. A fresh checkout does not include the raw data, processed splits, or saved model weights, so those artifacts must be generated locally before running the full app.

### 1. Add the source dataset

Place the authorized ISOT files at:

```text
data/raw/Fake.csv
data/raw/True.csv
```

### 2. Audit and prepare the data

Run from the repository root:

```powershell
python .\src\fake_news\data_audit.py
python .\src\fake_news\prepare_dataset.py
python .\src\fake_news\split_dataset.py
```

Review the console output and generated artifacts before continuing.

### 3. Train the TF-IDF baseline

```powershell
python .\src\fake_news\train_baseline.py
```

### 4. Train DistilBERT

```powershell
python .\src\fake_news\train_transformer.py
```

Transformer training can take significant time and memory. If the scripts or dependency versions change, consult their current source code and options before running them.

### 5. Evaluate and analyze

After the relevant models, processed splits, and reports exist, use the evaluation and analysis scripts in `src/fake_news/`, such as `evaluate_test.py` and `analyze_errors.py`. Keep final test results separate from decisions made during model tuning.

## Run the application

The application expects the locally generated or downloaded model artifacts and the reports/images it displays. After those dependencies are in place, run:

```powershell
python -m streamlit run .\app\app.py --server.fileWatcherType none
```

Open the local URL printed by Streamlit in your browser.

**Deployment note:** A hosted Streamlit instance starts in a fresh environment. Since the raw data and model weights are not stored in this repository, deployment requires a deliberate way to provision the required model artifacts and compatible dependencies. Do not assume that pushing this repository alone makes the app deployable.

## Run tests

From the repository root, with the virtual environment activated:

```powershell
python -m pytest -v
```

The development environment previously reported **8 passing tests** covering prediction behavior and CSV validation. Re-run the tests after changing dependencies or code; the recorded result is not a guarantee for every platform or fresh installation.

The GPU diagnostic is a standalone script, not a pytest test module.

## Technology stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Traditional machine learning:** scikit-learn, TF-IDF, Logistic Regression
- **Deep learning:** PyTorch, Hugging Face Transformers, DistilBERT
- **Explainability:** LIME
- **Visualization:** Matplotlib
- **Interactive app:** Streamlit
- **Testing:** pytest
- **Version control:** Git and GitHub

## Limitations and responsible use

1. **Headline-only inputs are insufficient for fact verification.** The model does not retrieve evidence or inspect the complete article.
2. **Dataset bias and leakage are possible.** Publisher, topic, event, and time-specific patterns may influence results.
3. **Confidence can be misleading.** A wrong answer can have a very high model score.
4. **External evaluation is limited.** The two-example challenge set is illustrative, not statistically representative.
5. **Generalization remains unproven.** Publisher-held-out, topic-held-out, and time-based evaluations would provide stronger evidence.
6. **Dataset rights matter.** Check the dataset's applicable terms before redistributing or using it commercially.

Do not use this project as the sole basis for labeling people, spreading allegations, making consequential decisions, or deciding whether a story is true. Verify claims through independent, credible sources.

## Future work

- Build a larger external benchmark with independently verified labels and evidence links.
- Evaluate on publisher-held-out and time-based splits.
- Investigate errors by topic and source without assuming causation from small samples.
- Evaluate probability calibration on a dedicated validation set.
- Explore retrieval-assisted verification that surfaces reliable supporting and contradicting evidence.
- Add continuous integration for tests and code quality checks.
- Improve artifact provisioning and document a reproducible deployment workflow.

## Repository and license

- **GitHub:** [Harshit765G4/fake-news-headline-classifier](https://github.com/Harshit765G4/fake-news-headline-classifier)
- **License:** No repository license is specified in this README. Choose and add a license only after confirming it is compatible with your code, dependencies, and the dataset's separate usage terms.
