import dask.dataframe as dd
import numpy as np
import pandas as pd
from utilities import print_ddf

df = pd.DataFrame(
    data={
        "A": list("ababab"),
        "B": np.arange(6),
    }
)
print(df, end="\n\n")

medians = df.groupby("A")["B"].median()
print("Medians:", end="\n\n")
print(medians.to_frame(), end="\n\n")

ddf = dd.from_pandas(df, npartitions=2)
print_ddf(ddf)

# Convert "A" and "B" to Pandas

medians = ddf[["A", "B"]].compute().groupby("A")["B"].median()
print("Medians:", end="\n\n")
print(medians, end="\n\n")

# No median method for SeriesGroupBy in Dask

ddf = ddf.set_index("A")
print_ddf(ddf)
medians = ddf.map_partitions(
    lambda partition_df: partition_df.groupby("A").median()
)
medians = medians.compute()
print("Medians:", end="\n\n")
print(medians, end="\n\n")
