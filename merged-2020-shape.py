import numpy as np
import pandas as pd

catalog_path = "/home/sila/etas/merged-2010-etas.csv"
shape_path = "/home/sila/etas/merged-2010-bbox-shape.npy"

df = pd.read_csv(catalog_path)
coords = df[["latitude", "longitude"]].dropna()

if coords.empty:
    raise ValueError("No rows have both latitude and longitude.")

lat_min, lat_max = coords["latitude"].min(), coords["latitude"].max()
lon_min, lon_max = coords["longitude"].min(), coords["longitude"].max()

print(f"Latitude:  {lat_min} to {lat_max}")
print(f"Longitude: {lon_min} to {lon_max}")

boundary = np.array([
    [lat_min, lon_min],
    [lat_max, lon_min],
    [lat_max, lon_max],
    [lat_min, lon_max],
    [lat_min, lon_min],
])
np.save(shape_path, boundary)
print(f"Saved ETAS boundary to {shape_path}")
