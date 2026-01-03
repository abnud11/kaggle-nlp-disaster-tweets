from collections import defaultdict
from typing import Any, cast
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
    pipeline,
)
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import evaluate
from torch.utils.data import Dataset
import numpy as np
import shap

set_seed(50)


def preprocess_data(df: pl.DataFrame, for_training: bool = True) -> pl.DataFrame:
    # Combine keyword, location, and text
    df = df.with_columns(
        combined_text=(
            pl.col("keyword").fill_null("")
            + pl.lit(" ")
            + pl.col("location").fill_null("")
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
        .str.replace_all(r"[^a-zA-Z0-9\s]", "")  # Remove special characters
        .str.replace_all(r"@\w+", "")  # Remove mentions
        .str.replace_all(r"\d", "")  # Remove digits
        .str.strip_chars()
    )

    return df.select(["combined_text", "target"]) if for_training else df.select(["combined_text"])


train_data = preprocess_data(pl.read_csv("train_split.csv"))
test_data = preprocess_data(pl.read_csv("test_split.csv"))


accuracy = evaluate.load("accuracy")
confusion_matrix = evaluate.load("confusion_matrix")
roc_auc_metric = evaluate.load("roc_auc")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

tran_model = "microsoft/xtremedistil-l6-h256-uncased"


class PolarsDataset(Dataset):
    def __init__(self, df: pl.DataFrame, tokenizer: PreTrainedTokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.row(idx, named=True)

        t_text = self.tokenizer(row["combined_text"], truncation=True, max_length=160)
        target = row.get("target")
        if target is not None:
            t_text["labels"] = row.get("target")
        return t_text


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
data_collator = DataCollatorWithPadding(return_tensors="pt", tokenizer=tokenizer, padding="max_length", max_length=160)
id2label = {0: "Neg", 1: "Pos"}
label2id = {"Neg": 0, "Pos": 1}
training_args = TrainingArguments(
    num_train_epochs=5,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    eval_strategy="epoch",
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    learning_rate=8e-5,
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


# SHAP Explanations for False Positives
# Create a pipeline for the explainer
explainer_pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    device=0 if model.device.type == "cuda" else -1,
)

# Initialize the SHAP explainer
explainer = shap.Explainer(explainer_pipe)

# Load false positives identified during evaluation
fp_df = pl.read_csv("false_positives.csv")
sample_fps = fp_df["False Positives"].head(10).to_list()
if sample_fps:
    print(f"Generating SHAP values for {len(sample_fps)} false positives...")
    shap_values = explainer(sample_fps)
    
    # This will render an interactive visualization in a Jupyter Notebook
    # In a standard script, it may require saving to HTML or using a notebook cell
    p = shap.plots.text(shap_values)
    with open("shap_false_positives.html", "w", encoding="utf-8") as f:
        f.write(f"<html><head>{shap.getjs()}</head><body>")
        f.write(shap.plots.text(shap_values, display=False))
        f.write("</body></html>")
"""
if sample_fps:
    print(f"Generating SHAP values for {len(sample_fps)} false positives...")
    shap_values = explainer(sample_fps)
    word_contributions = defaultdict(list)
    for i, val in enumerate(shap_values):
        # val.data يحتوي على الكلمات/التوكنز
        # val.values يحتوي على قيم SHAP المقابلة لها
        tokens = val.data
        scores = val.values[:, 1]
        
        for token, score in zip(tokens, scores):
            token = token.strip().lower()
            if score > 0: # نركز فقط على الكلمات التي دفعت النموذج نحو "إيجابي"
                word_contributions[token].append(score)

    # 5. تلخيص النتائج في جدول
    summary: list[dict[str, Any]] = []
    for token, scores in word_contributions.items():
        summary.append({
            'keyword': token,
            'appearance_count': len(scores),
            'average_impact': np.mean(scores),
            'total_impact': np.sum(scores)
        })

    # تحويل النتائج لـ DataFrame وترتيبها حسب التأثير الكلي
    result_df = pl.DataFrame(summary).sort(by='total_impact', descending=True)
    result_df.write_csv("false_positive_word_contributions.csv", quote_style="always")
"""


# Load false negatives identified during evaluation
fn_df = pl.read_csv("false_negatives.csv")
sample_fns = fn_df["False Negatives"].head(10).to_list()

if sample_fns:
    print(f"Generating SHAP values for {len(sample_fns)} false negatives...")
    shap_values = explainer(sample_fns)
    
    # This will render an interactive visualization in a Jupyter Notebook
    # In a standard script, it may require saving to HTML or using a notebook cell
    p = shap.plots.text(shap_values)
    with open("shap_false_negatives.html", "w", encoding="utf-8") as f:
        f.write(f"<html><head>{shap.getjs()}</head><body>")
        f.write(shap.plots.text(shap_values, display=False))
        f.write("</body></html>")

"""
kaggle_test_data = preprocess_data(pl.read_csv("test.csv"), for_training=False)
kaggle_tokenized_test_data = PolarsDataset(kaggle_test_data, tokenizer)
kaggle_predictions = trainer.predict(kaggle_tokenized_test_data)
kaggle_pred_labels = np.argmax(kaggle_predictions.predictions, axis=1)
submission_df = pl.DataFrame(
    {"id": pl.read_csv("test.csv")["id"], "target": kaggle_pred_labels})
submission_df.write_csv("submission.csv", quote_style="always")
"""