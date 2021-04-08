import pickle
from functools import wraps
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable, Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd


class Checkpoints:
    """Checkpoints class for persistent caching of data."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialise the checkpoint class.

        :param data_dir: Directory where to store the checkpoint data.
        """
        if data_dir is not None:
            self.dir = data_dir
        else:
            self.dir = Path(mkdtemp())
        self.dir.mkdir(exist_ok=True)

        self.checkpoint_dirs = [x for x in self.dir.iterdir() if x.is_dir()]
        self.current_checkpoint = 0

    def _get_checkpoint_dir(self, checkpoint: int) -> Path:
        """Get the checkpoint directory.

        :param checkpoint: Checkpoint number.
        :return: Path of the checkpoint directory.
        """
        return self.dir / f"checkpoint_{checkpoint:02d}"

    def _current_checkpoint_exists(self) -> bool:
        """Check if the current checkpoint exists.

        :return: True if the current checkpoint exists, else False.
        """
        current_checkpoint_dir = self._get_checkpoint_dir(
            self.current_checkpoint
        )
        return current_checkpoint_dir.exists()

    def _load_current_checkpoint(
        self,
    ) -> Union[Union[dd.DataFrame, Any], Tuple[Union[dd.DataFrame, Any], ...]]:
        """Load the current checkpoint.

        :return: Data items in the checkpoint.
        """
        checkpoint_dir = self._get_checkpoint_dir(self.current_checkpoint)
        files_in_checkpoint_dir = [x for x in checkpoint_dir.iterdir()]
        files_in_checkpoint_dir.sort(key=lambda file_: int(file_.stem[-2:]))

        data_items = list()
        for file in files_in_checkpoint_dir:
            if file.suffix == ".parquet":
                if file.is_dir():  # Dask
                    data_item_loaded = dd.read_parquet(file)

                    # Categorical columns and index from unknown to known
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

        self.current_checkpoint += 1

        if len(data_items) == 1:
            return data_items[0]
        else:
            return tuple(data_items)

    def _save_new_checkpoint(
        self, *data_items: Union[dd.DataFrame, pd.DataFrame, Any]
    ) -> Union[
        Union[dd.DataFrame, pd.DataFrame, Any],
        Tuple[Union[dd.DataFrame, pd.DataFrame, Any], ...],
    ]:
        """Save a new checkpoint.

        :param data_items: Data items to save in the checkpoint.
        """
        data_items_loaded = list()
        for i, data_item in enumerate(data_items):
            checkpoint_dir = self._get_checkpoint_dir(self.current_checkpoint)
            checkpoint_dir.mkdir(exist_ok=True)
            if isinstance(data_item, dd.DataFrame):
                file = checkpoint_dir / f"data_{i:02d}.parquet"

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
                file = checkpoint_dir / f"data_{i:02d}.parquet"
                data_item.to_parquet(file)
                data_item_loaded = pd.read_parquet(file)
                data_items_loaded.append(data_item_loaded)
            else:
                file = checkpoint_dir / f"data_{i:02d}.pkl"
                with file.open("wb") as f:
                    pickle.dump(data_item, f)
                with file.open("rb") as f:
                    data_item_loaded = pickle.load(f)
                data_items_loaded.append(data_item_loaded)

        self.current_checkpoint += 1

        if len(data_items_loaded) == 1:
            return data_items_loaded[0]
        else:
            return tuple(data_items_loaded)


def allow_checkpoint(callable: Callable) -> Callable:
    """Decorator to enable the checkpoint for a function or method.

    :param callable: Callable for which to allow a checkpoint.
    :return: Decorated callable.
    """

    @wraps(callable)
    def wrapper(*args, **kwargs):
        global _checkpoints

        if _checkpoints is not None:
            if _checkpoints._current_checkpoint_exists():
                checkpoint_data = _checkpoints._load_current_checkpoint()
            else:
                checkpoint_data = callable(*args, **kwargs)
                if not isinstance(checkpoint_data, tuple):
                    checkpoint_data = _checkpoints._save_new_checkpoint(
                        checkpoint_data
                    )
                else:
                    checkpoint_data = _checkpoints._save_new_checkpoint(
                        *checkpoint_data
                    )
        else:
            checkpoint_data = callable(*args, **kwargs)
        return checkpoint_data

    return wrapper


_checkpoints: Optional[Checkpoints] = None


def enable_checkpoints(data_dir: Optional[Path] = None) -> None:
    """Enable the checkpoints.

    :param data_dir: Path of the directory where to store the checkpoint data.
        If this argument is not specified, a temporary folder will be used.
    """
    global _checkpoints
    _checkpoints = Checkpoints(data_dir)
