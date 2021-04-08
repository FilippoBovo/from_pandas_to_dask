import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import List

import dask.dataframe as dd
import pandas as pd


class DaskDiskCache:
    def __init__(self):
        self.dir = Path(mkdtemp())
        self.dir.mkdir(exist_ok=True)

    def _cache_path(self, cache_name: str) -> Path:
        return self.dir / cache_name

    def persist(self, data: dd.DataFrame, cache_name: str) -> dd.DataFrame:
        cache_path = self._cache_path(cache_name)
        data.to_parquet(cache_path)
        data = dd.read_parquet(cache_path)

        # Categorical columns and index from unknown to known
        if isinstance(data.index.dtype, pd.CategoricalDtype):
            categorical_index = True
        else:
            categorical_index = False
        data = data.categorize(
            columns=data.select_dtypes("category").columns,
            index=categorical_index,
        )

        return data

    @property
    def cache_names(self) -> List[str]:
        return [x.name for x in self.dir.iterdir() if x.is_dir()]

    def remove(self, cache_name: str) -> None:
        if cache_name in self.cache_names:
            cache_path = self._cache_path(cache_name)
            shutil.rmtree(cache_path)
        else:
            raise Exception("Cache not found.")

    def cleanup(self) -> None:
        shutil.rmtree(self.dir)
