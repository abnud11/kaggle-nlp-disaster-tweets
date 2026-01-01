from sklearn.model_selection import train_test_split
import polars as pl
data = pl.read_csv("train.csv")

train_data, val_data = train_test_split(data.to_pandas(), test_size=0.5, random_state=42)

train_data.to_csv("train_split.csv", index=False)
val_data.to_csv("test_split.csv", index=False)
"""
data = pl.read_csv("train.csv")

print(f"Total samples: {data.height}")
print(f"Disaster messages: {data.filter(pl.col('target') == 1).height}")
print(f"Fake disaster messages: {data.filter(pl.col('target') == 0).height}")
"""