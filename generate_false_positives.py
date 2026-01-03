from typing import Any
import polars as pl

df = pl.read_csv("false_positive_word_contributions.csv")
df = df.sort('average_impact', descending=True).filter(pl.col('average_impact') > 0.02)


train_df = pl.read_csv("train.csv").select(['text', 'location', 'keyword', 'target']).unique('text')
generated_data = pl.read_csv("t.csv")
for col in generated_data.iter_columns():
    if col.name == "keyword":
        continue
    for i, sentence in enumerate(generated_data[col.name]):
        train_df = train_df.extend(pl.DataFrame({
            'text': [sentence],
            'location': [""],
            'keyword': [generated_data["keyword"][i]],
            "target": 0
        }).cast(train_df.schema))

train_df.write_csv("augmented_train.csv")

