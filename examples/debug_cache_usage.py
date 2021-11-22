from pathlib import Path

import dask.dataframe as dd
import debug_cache
import numpy as np
import pandas as pd


@debug_cache.cache
def create_ddf() -> dd.DataFrame:
    print("Creating DDF")

    df = pd.DataFrame(
        data={"A": np.arange(6), "B": np.array(list("xxyzyz"))},
        index=np.array(list("abcdef")),
    )
    ddf = dd.from_pandas(df, npartitions=2)
    ddf = ddf.categorize(columns=["B"], index=True)
    ddf["C"] = ddf["A"] ** 2

    return ddf


# Debug cache disabled

ddf = create_ddf()
print(f"Number of tasks: {len(ddf.dask)}\n")

# Debug cache enabled

debug_cache.enable(data_dir=Path("debug_cache_dir"))

ddf = create_ddf()
print(f"Number of tasks: {len(ddf.dask)}\n")
