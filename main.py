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
import datasets
import evaluate
import numpy as np

set_seed(850)
data = datasets.load_dataset(
    "csv", data_files={"train": "train_split.csv", "test": "test_split.csv"}
)
train_data = data["train"]
test_data = data["test"]
accuracy = evaluate.load("accuracy")
confusion_matrix = evaluate.load("confusion_matrix")
roc_auc_metric = evaluate.load("roc_auc")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

tran_model = "boltuix/bert-mini"

tokenizer = cast(
    PreTrainedTokenizer,
    AutoTokenizer.from_pretrained(tran_model),
)
def preprocess_function(examples):
    # Handle None values by replacing them with empty strings
    locations = [l if l is not None else "" for l in examples["location"]]
    keywords = [k if k is not None else "" for k in examples["keyword"]]
    
    t_text = tokenizer(examples["text"], truncation=True, add_special_tokens=False)
    t_location = tokenizer(locations, truncation=True, add_special_tokens=False)
    t_keyword = tokenizer(keywords, truncation=True, add_special_tokens=False)
    
    batch_input_ids = []
    for i in range(len(examples["text"])):
        # Concatenate input_ids for each example in the batch
        combined_ids = t_text["input_ids"][i] + t_location["input_ids"][i] + t_keyword["input_ids"][i]
        batch_input_ids.append(tokenizer.build_inputs_with_special_tokens(combined_ids))
        
    return {"input_ids": batch_input_ids, "labels": examples["target"]}


def compute_metrics(eval_pred: EvalPrediction):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    false_positives = [
        test_data["text"][i]
        for i, (true, pred) in enumerate(zip(labels, predictions))
        if true == 0 and pred == 1
    ]
    false_negatives = [
        test_data["text"][i]
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



tokenized_train_data = train_data.map(preprocess_function, batched=True)
tokenized_test_data = test_data.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(return_tensors="pt", tokenizer=tokenizer)
id2label = {0: "Neg", 1: "Pos"}
label2id = {"Neg": 0, "Pos": 1}
training_args = TrainingArguments(
    num_train_epochs=4,
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
