import numpy as np
import pandas as pd
import pytest
from moneyed import USD, Money

import pandas_money as pm


@pytest.fixture
def money_dict():
    return {"a": Money("3.14", USD), "b": Money("2.72", USD)}


@pytest.fixture
def multi_tuples():
    return [("x", "a"), ("x", "b"), ("y", "a"), ("y", "b")]


@pytest.fixture
def money_multidict(multi_tuples):
    return {idx: np.random.randint(1, 100) for idx in multi_tuples}


@pytest.fixture
def money_series(money_dict):
    return pm.money_series(money_dict)


def test_loc(money_dict, money_series):
    assert money_series.loc["a"] == money_dict["a"]
    assert money_series.loc["b"] == money_dict["b"]


def test_iloc(money_dict, money_series):
    assert money_series.iloc[0] == money_dict["a"]
    assert money_series.iloc[1] == money_dict["b"]


def test_multiindex_loc(multi_tuples, money_dict):
    index = pd.MultiIndex.from_tuples(multi_tuples, names=["first", "second"])
    values = list(money_dict.values()) + list(money_dict.values())
    s = pd.Series(values, index=index, dtype="Money64")
    for idx, val in zip(multi_tuples, values):
        assert s.loc[idx] == val


def test_set_loc(money_dict):
    index = list(money_dict.keys())
    s = pd.Series(index=index, dtype="Money64")
    assert all(list(pd.isna(s)))
    for idx in index:
        s.loc[idx] = money_dict[idx]
    for idx in index:
        assert s.loc[idx] == money_dict[idx]


def test_set_iloc(money_dict):
    index = list(money_dict.keys())
    s = pd.Series(index=index, dtype="Money64")
    assert all(list(pd.isna(s)))
    for i, idx in enumerate(index):
        s.iloc[i] = money_dict[idx]
    for i, idx in enumerate(index):
        assert s.iloc[i] == money_dict[idx]


@pytest.mark.skip(reason="not supported yet")
def test_no_data_isha(money_dict):
    index = list(money_dict.keys())
    s = pm.series(index=index)
    assert all(list(pd.isna(s)))
