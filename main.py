from typing import cast
import polars as pl
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import evaluate
from torch.utils.data import Dataset
import numpy as np

set_seed(850)


def preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    # Combine keyword, location, and text
    df = df.with_columns(
        combined_text=(
            pl.lit("This is a Twitter tweet") + pl.when(pl.col("keyword").fill_null("") != "")
            .then(pl.lit(", the keyword is ") + pl.col("keyword") + pl.lit(","))
            .otherwise(pl.lit(""))
            + pl.lit(" ")
            + pl.when(pl.col("location").fill_null("") != "")
            .then(pl.lit(", it was posted by a person located at ") + pl.col("location") + pl.lit(","))
            .otherwise(pl.lit(""))
            + pl.lit(" ")
            + pl.col("text").fill_null("")
        )
    )

    # Basic preprocessing using Polars string methods
    df = df.with_columns(
        combined_text=pl.col("combined_text")
        .str.to_lowercase()
        .str.replace_all(r"http\S+|www\S+|https\S+", "")  # Remove URLs
        .str.replace_all(r"\s+", " ")  # Remove extra whitespace
        .str.replace_all(r"%20", " ")  # Replace URL encoded spaces
        .str.strip_chars()
    )

    return df.select(["combined_text", "target"])


train_data = preprocess_data(pl.read_csv("train_split.csv"))
test_data = preprocess_data(pl.read_csv("test_split.csv"))


accuracy = evaluate.load("accuracy")
confusion_matrix = evaluate.load("confusion_matrix")
roc_auc_metric = evaluate.load("roc_auc")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

tran_model = "boltuix/bert-mini"


class PolarsDataset(Dataset):
    def __init__(self, df: pl.DataFrame, tokenizer: PreTrainedTokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.row(idx, named=True)

        t_text = self.tokenizer(row["combined_text"])
        return {"input_ids": t_text["input_ids"], "labels": row["target"]}


tokenizer = cast(
    PreTrainedTokenizer,
    AutoTokenizer.from_pretrained(tran_model),
)


def compute_metrics(eval_pred: EvalPrediction):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    false_positives = [
        test_data["combined_text"][i]
        for i, (true, pred) in enumerate(zip(labels, predictions))
        if true == 0 and pred == 1
    ]
    false_negatives = [
        test_data["combined_text"][i]
        for i, (true, pred) in enumerate(zip(labels, predictions))
        if true == 1 and pred == 0
    ]
    pl.DataFrame({"False Positives": false_positives}).write_csv(
        "false_positives.csv", quote_style="always"
    )
    pl.DataFrame({"False Negatives": false_negatives}).write_csv(
        "false_negatives.csv", quote_style="always"
    )
    cm = confusion_matrix.compute(predictions=predictions, references=labels)
    cm["confusion_matrix"] = cm["confusion_matrix"].tolist()
    accuracy_score = accuracy.compute(predictions=predictions, references=labels)
    roc_auc_score = roc_auc_metric.compute(
        prediction_scores=predictions, references=labels
    )
    f1_metric_score = f1_metric.compute(predictions=predictions, references=labels)
    precision_score = precision_metric.compute(
        predictions=predictions, references=labels
    )
    recall_score = recall_metric.compute(predictions=predictions, references=labels)
    return {
        **cm,
        **accuracy_score,
        **roc_auc_score,
        **f1_metric_score,
        **precision_score,
        **recall_score,
    }


tokenized_train_data = PolarsDataset(train_data, tokenizer)
tokenized_test_data = PolarsDataset(test_data, tokenizer)
data_collator = DataCollatorWithPadding(return_tensors="pt", tokenizer=tokenizer)
id2label = {0: "Neg", 1: "Pos"}
label2id = {"Neg": 0, "Pos": 1}
training_args = TrainingArguments(
    num_train_epochs=5,
    eval_strategy="epoch",
)
model = cast(
    PreTrainedModel,
    AutoModelForSequenceClassification.from_pretrained(
        tran_model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    ),
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy'] * 100:.2f}%")
print(f"ROC AUC Score: {eval_results['eval_roc_auc']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")
ConfusionMatrixDisplay(
    confusion_matrix=np.array(eval_results["eval_confusion_matrix"]),
    display_labels=["Neg", "Pos"],
).plot()
plt.show()
