import checkpoints
import dask.dataframe as dd
import numpy as np
import pandas as pd


@checkpoints.allow_checkpoint
def create_dask_dataframe() -> dd.DataFrame:
    df = pd.DataFrame(
        data={"A": np.arange(6), "B": np.array(list("xxyzyz"))},
        index=np.array(list("abcdef")),
    )
    ddf = dd.from_pandas(df, npartitions=2)
    ddf = ddf.categorize(columns=["B"], index=True)
    ddf["C"] = ddf["A"] ** 2

    return ddf


# Checkpoints disabled

ddf = create_dask_dataframe()
print(f"Number of tasks without checkpoint: {len(ddf.dask)}")

# Checkpoints enabled

checkpoints.enable_checkpoints()

ddf = create_dask_dataframe()
print(f"First call -> Number of tasks with checkpoint: {len(ddf.dask)}")

ddf = create_dask_dataframe()
print(f"Second call -> Number of tasks with checkpoint: {len(ddf.dask)}")
