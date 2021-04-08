import dask.dataframe as dd
import numpy as np
import pandas as pd
from utilities import print_ddf

df = pd.DataFrame(
    data={
        "A": np.arange(6),
        "B": np.array([2, 3, 0, 4, 1, 5]),
        "C": np.array(list("abcdef")),
        "D": np.array(list("fbacde")),
        "E": np.array(list("aabaab")),
    }
)

print("Original Dask DataFrame:", end="\n\n")
ddf = dd.from_pandas(df, npartitions=2)
print_ddf(ddf)

print("Reset the index:", end="\n\n")
ddf = dd.from_pandas(df, npartitions=2).reset_index()
print_ddf(ddf)

print("Set integer index:", end="\n\n")
ddf = dd.from_pandas(df, npartitions=2).set_index("B")
print_ddf(ddf)

print("Set sorted string index:", end="\n\n")
ddf = dd.from_pandas(df, npartitions=2).set_index("C")
print_ddf(ddf)

print("Set unsorted string index:", end="\n\n")
ddf = dd.from_pandas(df, npartitions=2).set_index("D")
print_ddf(ddf)

print("Set unsorted string index with duplicates:", end="\n\n")
ddf = dd.from_pandas(df, npartitions=2).set_index("E")
print_ddf(ddf)

print("Set unsorted string index with duplicates (3 partitions):", end="\n\n")
ddf = dd.from_pandas(df, npartitions=3).set_index("E")
print_ddf(ddf)
