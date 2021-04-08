import dask.dataframe as dd
import numpy as np
import pandas as pd
from utilities import print_ddf

df = pd.DataFrame(
    data={"A": np.arange(6)},
    index=pd.Index(np.array(list("ababab")), name="index"),
)

print("Starting Dask DataFrame with 1 partition:", end="\n\n")
ddf = dd.from_pandas(df, npartitions=1)
print_ddf(ddf)

print("Repartition with 3 partitions (no index isolation):", end="\n\n")
ddf = dd.from_pandas(df, npartitions=1).repartition(npartitions=3)
print_ddf(ddf)

print("Repartition with 3 partitions (index isolation):", end="\n\n")
ddf = (
    dd.from_pandas(df, npartitions=1)
    .reset_index()
    .set_index("index", npartitions=3)
)
print_ddf(ddf)

print(
    "Repartition with 3 partitions (index isolation alternative):", end="\n\n"
)
ddf = (
    dd.from_pandas(df, npartitions=1)
    .reset_index()
    .repartition(npartitions=3)
    .set_index("index")
)
print_ddf(ddf)
