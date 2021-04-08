import unittest

import dask.dataframe as dd
import numpy as np
import pandas as pd


def mse(ddf, column_a, column_b):
    return ((ddf[column_a] - ddf[column_b]) ** 2).mean()


class TestMse(unittest.TestCase):
    def test(self):
        df = pd.DataFrame(
            data={
                "A": [2.0, 5.0, 2.0],
                "B": [1.0, 5.0, 4.0],
            }
        )

        expected_output = 1.6666666666666667

        for npartitions in [1, 2, 3]:
            with self.subTest(npartitions=npartitions):
                ddf = dd.from_pandas(df, npartitions=npartitions)
                output = mse(ddf, "A", "B").compute()
                self.assertEqual(output, expected_output)


unittest.main()
