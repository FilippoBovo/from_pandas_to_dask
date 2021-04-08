from time import time

import dask.dataframe as dd
import numpy as np
import pandas as pd

df = pd.DataFrame(
    data={
        "a": np.random.permutation(100_000),
        "b": np.random.random(100_000),
        "c": np.random.random(100_000),
    }
)

print("Pandas DataFrame head:")
print(df.head(), end="\n\n")

ddf = dd.from_pandas(df, npartitions=20)
print(ddf, end="\n\n")
print("Dask DataFrame head:")
print(ddf.head(), end="\n\n")


print("\nSum columns:")

start_time = time()
df["b"] + df["c"]
end_time = time()
print(f"  Pandas DataFrame: {(end_time - start_time) * 1000:7.3f} ms")

start_time = time()
(ddf["b"] + ddf["c"]).compute()
end_time = time()
print(f"  Dask DataFrame:   {(end_time - start_time) * 1000:7.3f} ms")


print("\nSet index:")

start_time = time()
df.set_index("a")
end_time = time()
print(f"  Pandas DataFrame: {(end_time - start_time) * 1000:7.3f} ms")

start_time = time()
ddf.set_index("a").compute()
end_time = time()
print(f"  Dask DataFrame:   {(end_time - start_time) * 1000:7.3f} ms")
