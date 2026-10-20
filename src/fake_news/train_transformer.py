
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "distilbert_fake_news"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 64
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
SEED = 42

LABEL_NAMES = ["Fake", "Real"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HeadlineDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.headlines = frame["headline"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.headlines[index],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index], dtype=torch.long
        )
        return item


def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            batch_size = batch["labels"].size(0)

            total_loss += outputs.loss.item() * batch_size
            total_examples += batch_size

            predictions = outputs.logits.argmax(dim=1)

            all_labels.extend(batch["labels"].cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision_macro": precision_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "labels": all_labels,
        "predictions": all_predictions,
    }


def main():
    set_seed(SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DISTILBERT FAKE NEWS HEADLINE CLASSIFIER")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    train_frame = pd.read_csv(DATA_DIR / "train.csv")
    val_frame = pd.read_csv(DATA_DIR / "validation.csv")

    print(f"Training rows: {len(train_frame):,}")
    print(f"Validation rows: {len(val_frame):,}")
    print(f"Maximum sequence length: {MAX_LENGTH}")
    print(f"Training batch size: {TRAIN_BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    model.to(DEVICE)

    train_dataset = HeadlineDataset(train_frame, tokenizer)
    val_dataset = HeadlineDataset(val_frame, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(
            0.0, 1.0 - step / max(total_steps, 1)
        ),
    )

    best_f1 = -1.0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        examples_seen = 0

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for step, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            batch_size = batch["labels"].size(0)
            running_loss += loss.item() * batch_size
            examples_seen += batch_size

            if step % 250 == 0 or step == len(train_loader):
                print(
                    f"  Batch {step}/{len(train_loader)}"
                    f" | mean training loss: "
                    f"{running_loss / examples_seen:.4f}"
                )

        validation = evaluate(model, val_loader)

        epoch_result = {
            "epoch": epoch + 1,
            "training_loss": running_loss / examples_seen,
            "validation_loss": validation["loss"],
            "accuracy": validation["accuracy"],
            "precision_macro": validation["precision_macro"],
            "recall_macro": validation["recall_macro"],
            "f1_macro": validation["f1_macro"],
        }
        history.append(epoch_result)

        print("\nValidation metrics:")
        for key in (
            "validation_loss",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ):
            print(f"  {key}: {epoch_result[key]:.4f}")

        if validation["f1_macro"] > best_f1:
            best_f1 = validation["f1_macro"]

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            best_predictions = validation["predictions"]
            best_labels = validation["labels"]
            best_epoch = epoch + 1

            print(f"  Saved best model from epoch {best_epoch}")

        if torch.cuda.is_available():
            print(
                "  Peak allocated GPU memory: "
                f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
            )

    # Report metrics for the best validation checkpoint.
    report = {
        "model": MODEL_NAME,
        "seed": SEED,
        "device": str(DEVICE),
        "max_length": MAX_LENGTH,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "best_epoch": best_epoch,
        "training_rows": len(train_frame),
        "validation_rows": len(val_frame),
        "best_validation_metrics": {
            "accuracy": accuracy_score(best_labels, best_predictions),
            "precision_macro": precision_score(
                best_labels, best_predictions,
                average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                best_labels, best_predictions,
                average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                best_labels, best_predictions,
                average="macro", zero_division=0
            ),
            "classification_report": classification_report(
                best_labels,
                best_predictions,
                labels=[0, 1],
                target_names=LABEL_NAMES,
                output_dict=True,
                zero_division=0,
            ),
        },
        "history": history,
    }

    metrics_path = REPORT_DIR / "transformer_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(
        best_labels, best_predictions, labels=[0, 1]
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABEL_NAMES,
    )
    display.plot(cmap="viridis", values_format="d")
    plt.title("DistilBERT Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "transformer_confusion_matrix.png",
        dpi=160,
    )
    plt.close()

    print("\n" + "=" * 60)
    print("BEST CHECKPOINT SUMMARY")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    for key, value in report["best_validation_metrics"].items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")

    print(f"\nSaved model: {MODEL_DIR}")
    print(f"Saved metrics: {metrics_path}")
    print(
        "Saved confusion matrix: "
        f"{FIGURE_DIR / 'transformer_confusion_matrix.png'}"
    )
    print("\nThe test set has not been loaded or evaluated.")


if __name__ == "__main__":
    main()
