import pandas as pd

input_file = "/home/sila/etas/merged-2010-catalog.csv"
output_file = "/home/sila/etas/merged-2010-etas.csv"

df = pd.read_csv(input_file)

required_columns = {"index", "event_time", "latitude", "longitude", "magnitude"}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df.rename(columns={"index": "id", "event_time": "time"})
df = df[["id", "latitude", "longitude", "time", "magnitude"]]
df["time"] = pd.to_datetime(df["time"], errors="raise")

if df.isna().any().any():
    raise ValueError("The converted catalog contains missing values.")

df.to_csv(output_file, index=False)
print(f"Saved {len(df)} events to {output_file}")
