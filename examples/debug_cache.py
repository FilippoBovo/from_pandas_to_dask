import pickle
from functools import wraps
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable, Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd


class DebugCache:
    """Debugging class for the persistent caching of data."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialise the debug cache class.

        :param base_dir: Directory where to cache the data.
        """
        if base_dir is not None:
            self.base_dir = base_dir
        else:
            self.base_dir = Path(mkdtemp())
        self.base_dir.mkdir(exist_ok=True)

        self.cache_dirs = [x for x in self.base_dir.iterdir() if x.is_dir()]
        self.current_cache_index = 0

    def _get_cache_dir(self, index: int) -> Path:
        """Get a cache directory by index.

        :param cache: Cache index.
        :return: Path of the cache directory.
        """
        return self.base_dir / f"cache_{index:02d}"

    def _current_cache_exists(self) -> bool:
        """Check if the current cache exists.

        :return: True if the current cache exists, else False.
        """
        current_cache_dir = self._get_cache_dir(self.current_cache_index)
        return current_cache_dir.exists()

    def _load_current_cache(
        self,
    ) -> Union[Union[dd.DataFrame, Any], Tuple[Union[dd.DataFrame, Any], ...]]:
        """Load the current cache.

        :return: Data items in the cache.
        """
        cache_dir = self._get_cache_dir(self.current_cache_index)
        files_in_cache_dir = [x for x in cache_dir.iterdir()]
        files_in_cache_dir.sort(key=lambda file_: int(file_.stem[-2:]))

        data_items = list()
        for file in files_in_cache_dir:
            if file.suffix == ".parquet":
                if file.is_dir():  # Dask
                    data_item_loaded = dd.read_parquet(file)

                    # Categorical columns and index from unknown to known
                    # (https://docs.dask.org/en/latest/dataframe-categoricals.html)
                    if isinstance(
                        data_item_loaded.index.dtype, pd.CategoricalDtype
                    ):
                        categorical_index = True
                    else:
                        categorical_index = False
                    data_item_loaded = data_item_loaded.categorize(
                        columns=data_item_loaded.select_dtypes(
                            "category"
                        ).columns,
                        index=categorical_index,
                    )

                    data_items.append(data_item_loaded)
                else:  # Pandas
                    data_item_loaded = pd.read_parquet(file)
                    data_items.append(data_item_loaded)
            else:  # Any other data type
                with file.open("rb") as f:
                    data_item = pickle.load(f)
                data_items.append(data_item)

        self.current_cache_index += 1

        if len(data_items) == 1:
            return data_items[0]
        else:
            return tuple(data_items)

    def _cache_new_data(
        self, *data_items: Union[dd.DataFrame, pd.DataFrame, Any]
    ) -> Union[
        Union[dd.DataFrame, pd.DataFrame, Any],
        Tuple[Union[dd.DataFrame, pd.DataFrame, Any], ...],
    ]:
        """Cache new data.

        :param data_items: Data items to cache.
        """
        data_items_loaded = list()
        for i, data_item in enumerate(data_items):
            cache_dir = self._get_cache_dir(self.current_cache_index)
            cache_dir.mkdir(exist_ok=True)
            if isinstance(data_item, dd.DataFrame):
                file = cache_dir / f"data_{i:02d}.parquet"

                data_item.to_parquet(file)
                data_item_loaded = dd.read_parquet(file)

                # Categorical columns and index from unknown to known
                if isinstance(
                    data_item_loaded.index.dtype, pd.CategoricalDtype
                ):
                    categorical_index = True
                else:
                    categorical_index = False
                data_item_loaded = data_item_loaded.categorize(
                    columns=data_item_loaded.select_dtypes("category").columns,
                    index=categorical_index,
                )

                data_items_loaded.append(data_item_loaded)
            elif isinstance(data_item, pd.DataFrame):
                file = cache_dir / f"data_{i:02d}.parquet"
                data_item.to_parquet(file)
                data_item_loaded = pd.read_parquet(file)
                data_items_loaded.append(data_item_loaded)
            else:
                file = cache_dir / f"data_{i:02d}.pkl"
                with file.open("wb") as f:
                    pickle.dump(data_item, f)
                with file.open("rb") as f:
                    data_item_loaded = pickle.load(f)
                data_items_loaded.append(data_item_loaded)

        self.current_cache_index += 1

        if len(data_items_loaded) == 1:
            return data_items_loaded[0]
        else:
            return tuple(data_items_loaded)


def cache(callable: Callable) -> Callable:
    """Decorator to enable caching for a function or method.

    :param callable: Function or method for which to enable caching.
    :return: Decorated function or method.
    """

    @wraps(callable)
    def wrapper(*args, **kwargs):
        global _caches

        if _caches is not None:
            if _caches._current_cache_exists():
                cache_data = _caches._load_current_cache()
            else:
                cache_data = callable(*args, **kwargs)
                if not isinstance(cache_data, tuple):
                    cache_data = _caches._cache_new_data(cache_data)
                else:
                    cache_data = _caches._cache_new_data(*cache_data)
        else:
            cache_data = callable(*args, **kwargs)
        return cache_data

    return wrapper


_caches: Optional[DebugCache] = None


def enable(data_dir: Optional[Path] = None) -> None:
    """Enable debug caching.

    :param data_dir: Path of the directory where to cache the data.
        If this argument is not specified, a temporary directory will be used.
    """
    global _caches
    _caches = DebugCache(data_dir)
