import shutil
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from disk_cache import DaskDiskCache
from utilities import print_ddf

df = pd.DataFrame(
    data={"A": np.arange(6), "B": np.array(list("xxyzyz"))},
    index=np.array(list("abcdef")),
)
ddf = dd.from_pandas(df, npartitions=2)
ddf = ddf.categorize(columns=["B"], index=True)
ddf["C"] = ddf["A"] ** 2

print_ddf(ddf, end="\n\n")
print(ddf, end="\n\n")


# Simple persistance

data_dir = Path("data.parquet")
ddf.to_parquet(data_dir)
ddf_cached = dd.read_parquet(data_dir)

print_ddf(ddf_cached, end="\n\n")
print(ddf_cached, end="\n\n")

shutil.rmtree(data_dir)


# DaskDiskCache utility class

dask_disk_cache = DaskDiskCache()
ddf = dask_disk_cache.persist(ddf, cache_name="data")

print(f"Cache names: {dask_disk_cache.cache_names}", end="\n\n")

print_ddf(ddf, end="\n\n")
print(ddf, end="\n\n")

dask_disk_cache.remove(cache_name="data")
dask_disk_cache.cleanup()
