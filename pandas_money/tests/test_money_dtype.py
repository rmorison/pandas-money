import contextlib
import io
import re
import string
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd
import pytest
from moneyed import USD, Money

import pandas_money as pm

# Number of digits after decimal we check on ratios
ratio_precision = 4


@pytest.fixture
def money_dict():
    return {"a": Money("3.14", USD), "b": Money("2.72", USD)}


@pytest.fixture
def money_list(money_dict):
    return list(money_dict.values())


@pytest.fixture
def money_scalar():
    return Money("3.14", USD)


@pytest.fixture
def decimal_list():
    return [Decimal("3.14"), Decimal("2.72")]


@pytest.fixture
def decimal_scalar():
    return Decimal("3.14")


@pytest.fixture
def float_list():
    return [3.14, 2.72]


@pytest.fixture
def float_scalar():
    return 3.14


@pytest.fixture
def str_scalar():
    return "3.14"


@pytest.fixture
def int_scalar():
    return 314


@pytest.fixture
def slice_series():
    return pd.Series(
        {label: i for i, label in enumerate(string.ascii_lowercase[:10])},
        dtype="Money64",
    )


def test_money_dict(money_dict):
    m = pd.Series(money_dict, dtype="Money64")
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m["a"] == money_dict["a"]
    assert m["b"] == money_dict["b"]


def test_money_list(money_list):
    m = pd.Series(money_list, dtype="Money64")
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == money_list[0]
    assert m.iloc[1] == money_list[1]


def test_money_scalar(money_scalar):
    m = pd.Series(money_scalar, dtype="Money64")
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == money_scalar


def test_money_series_in_dataframe(money_scalar):
    df = pd.DataFrame(index=range(2))
    df["m"] = pm.money_series(money_scalar)
    assert isinstance(df["m"].dtype, pm.MoneyDtype)
    assert df["m"].iloc[0] == money_scalar
    assert pd.isna(df["m"].iloc[1])


def test_decimal_scalar(decimal_scalar):
    m = pd.Series(decimal_scalar, dtype="Money64")
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == Money(decimal_scalar, USD)


def test_str_scalar(str_scalar):
    m = pd.Series(str_scalar, dtype="Money64")
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == Money(str_scalar, USD)


def test_int_scalar(int_scalar):
    m = pd.Series(int_scalar, dtype="Money64")
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == Money(int_scalar, USD)


def test_rounding():
    down = Decimal("1.444999")
    up = Decimal("1.445000")
    m = pd.Series([down, up], dtype="Money64")
    assert m.iloc[0] == Money(down.quantize(Decimal(".01")), USD)
    assert m.iloc[1] == Money(up.quantize(Decimal(".01")), USD)


def test_add_money(money_list, money_scalar):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = money_scalar
    m = m1 + m2
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == money_list[0] + money_scalar
    assert m.iloc[1] == money_list[1] + money_scalar


def test_add_money_array(money_list):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = pd.Series(money_list, dtype="Money64")
    m = m1 + m2
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == money_list[0] + money_list[0]
    assert m.iloc[1] == money_list[1] + money_list[1]


def test_sub_money(money_list, money_scalar):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = money_scalar
    m = m1 - m2
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == money_list[0] - money_scalar
    assert m.iloc[1] == money_list[1] - money_scalar


def test_sub_money_array(money_list):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = pd.Series(money_list, dtype="Money64")
    m = m1 - m2
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == money_list[0] - money_list[0]
    assert m.iloc[1] == money_list[1] - money_list[1]


def test_mul_money_array_money_array(money_list):
    m1 = pd.Series(money_list, dtype="Money64")
    with pytest.raises(TypeError):
        m1 * m1


def test_mul_money_array_money(money_list, money_scalar):
    m1 = pd.Series(money_list, dtype="Money64")
    with pytest.raises(TypeError):
        m1 * money_scalar


def test_mul_money_array_decimal_array(money_list, decimal_list):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = pd.Series(decimal_list)
    m = m1 * m2
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == (money_list[0] * decimal_list[0]).round(pm.money_decimals)
    assert m.iloc[1] == (money_list[1] * decimal_list[1]).round(pm.money_decimals)


def test_mul_money_array_decimal(money_list, decimal_scalar):
    m1 = pd.Series(money_list, dtype="Money64")
    m = m1 * decimal_scalar
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == (money_list[0] * decimal_scalar).round(pm.money_decimals)
    assert m.iloc[1] == (money_list[1] * decimal_scalar).round(pm.money_decimals)


def test_mul_money_array_float_array(money_list, float_list):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = pd.Series(float_list)
    m = m1 * m2
    assert isinstance(m.dtype, pm.MoneyDtype)
    # DeprecationWarning: Multiplying Money instances with floats is deprecated
    # so cast floats to Decimal
    assert m.iloc[0] == (money_list[0] * Decimal(float_list[0])).round(
        pm.money_decimals
    )
    assert m.iloc[1] == (money_list[1] * Decimal(float_list[1])).round(
        pm.money_decimals
    )


def test_mul_money_array_float(money_list, float_scalar):
    m1 = pd.Series(money_list, dtype="Money64")
    m = m1 * float_scalar
    assert isinstance(m.dtype, pm.MoneyDtype)
    # DeprecationWarning: Multiplying Money instances with floats is deprecated
    # so cast floats to Decimal
    assert m.iloc[0] == (money_list[0] * Decimal(float_scalar)).round(pm.money_decimals)
    assert m.iloc[1] == (money_list[1] * Decimal(float_scalar)).round(pm.money_decimals)


def test_truediv_money_array_money_array(money_list):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = pd.Series(reversed(money_list), dtype="Money64")
    m = m1 / m2
    assert isinstance(m.dtype, pd.Float64Dtype)
    assert round(m.iloc[0], ratio_precision) == round(
        float(money_list[0].amount) / float(money_list[1].amount),
        ratio_precision,
    )
    assert round(m.iloc[1], ratio_precision) == round(
        float(money_list[1].amount) / float(money_list[0].amount),
        ratio_precision,
    )


def test_truediv_money_array_money(money_list, money_scalar):
    m1 = pd.Series(money_list, dtype="Money64")
    m = m1 / money_scalar
    assert isinstance(m.dtype, pd.Float64Dtype)
    assert round(m.iloc[0], ratio_precision) == round(
        float(money_list[0].amount) / float(money_scalar.amount),
        ratio_precision,
    )
    assert round(m.iloc[1], ratio_precision) == round(
        float(money_list[1].amount) / float(money_scalar.amount),
        ratio_precision,
    )


def test_truediv_money_array_decimal_array(money_list, decimal_list):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = pd.Series(reversed(decimal_list))
    m = m1 / m2
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == (money_list[0] / decimal_list[1]).round(pm.money_decimals)
    assert m.iloc[1] == (money_list[1] / decimal_list[0]).round(pm.money_decimals)


def test_truediv_money_array_decimal(money_list, decimal_scalar):
    m1 = pd.Series(money_list, dtype="Money64")
    m = m1 / decimal_scalar
    assert isinstance(m.dtype, pm.MoneyDtype)
    assert m.iloc[0] == (money_list[0] / decimal_scalar).round(pm.money_decimals)
    assert m.iloc[1] == (money_list[1] / decimal_scalar).round(pm.money_decimals)


def test_truediv_money_array_float_array(money_list, float_list):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = pd.Series(reversed(float_list))
    m = m1 / m2
    assert isinstance(m.dtype, pm.MoneyDtype)
    # DeprecationWarning: Multiplying Money instances with floats is deprecated
    # so cast floats to Decimal
    assert m.iloc[0] == (money_list[0] / Decimal(float_list[1])).round(
        pm.money_decimals
    )
    assert m.iloc[1] == (money_list[1] / Decimal(float_list[0])).round(
        pm.money_decimals
    )


def test_truediv_money_array_float(money_list, float_scalar):
    m1 = pd.Series(money_list, dtype="Money64")
    m = m1 / float_scalar
    assert isinstance(m.dtype, pm.MoneyDtype)
    # DeprecationWarning: Multiplying Money instances with floats is deprecated
    # so cast floats to Decimal
    assert m.iloc[0] == (money_list[0] / Decimal(float_scalar)).round(pm.money_decimals)
    assert m.iloc[1] == (money_list[1] / Decimal(float_scalar)).round(pm.money_decimals)


def test_truediv_by_zero(money_list):
    m1 = pm.money_series(money_list)
    m = m1 / [0, 1]
    assert pd.isna(m.iloc[0])
    assert m.iloc[1] == money_list[1]


def test_badscalar():
    with pytest.raises(InvalidOperation):
        pd.Series("asdf", dtype="Money64")


def test_astype(money_list):
    m1 = pd.Series(money_list, dtype="Money64")
    m = m1.astype("Float64")
    assert isinstance(m.dtype, pd.Float64Dtype)
    assert round(m.iloc[0], pm.money_decimals) == round(
        float(money_list[0].amount),
        pm.money_decimals,
    )
    assert round(m.iloc[1], pm.money_decimals) == round(
        float(money_list[1].amount),
        pm.money_decimals,
    )


def test_eq(money_list):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = pd.Series(reversed(money_list), dtype="Money64")
    assert list(m1 == m1) == [True, True]
    assert list(m1 == pd.Series(money_list, dtype="Money64")) == [True, True]
    assert list(m1 != m2) == [True, True]


def test_eq_not_money(money_list):
    m = pd.Series(money_list, dtype="Money64")
    assert list(m == money_list) == [True, True]


def test_bad_money_type():
    with pytest.raises(TypeError):
        pd.Series(object(), dtype="Money64")


def test_slice_head(slice_series):
    slen = len(slice_series)
    s = slice_series[: slen // 2]
    assert len(s) == slen // 2
    for i, item in enumerate(s):
        assert slice_series.iloc[i] == item


def test_slice_skip(slice_series):
    slen = len(slice_series)
    s = slice_series[::2]
    assert len(s) == slen // 2
    for i, item in enumerate(s):
        assert slice_series.iloc[2 * i] == item


def test_slice_reverse(slice_series):
    slen = len(slice_series)
    s = slice_series[::-1]
    assert len(s) == slen
    for i, item in enumerate(s):
        assert slice_series.iloc[slen - i - 1] == item


def test_slice_index(slice_series):
    select = ["a", "d", "e"]
    s = slice_series[select]
    assert len(s) == len(select)
    for idx, v in zip(select, s):
        assert slice_series[idx] == v


def test_from_strings():
    strings = "3.14 2.72".split()
    m = pd.Series(strings, dtype="Money64")
    assert m.iloc[0] == Money(strings[0], USD)
    assert m.iloc[1] == Money(strings[1], USD)


def test_nan():
    m = pd.Series(np.nan, dtype="Money64")
    assert pd.isna(m.iloc[0])


def test_reindex(money_list):
    m = pd.Series(money_list, dtype="Money64").reindex(range(len(money_list) + 1))
    assert not any(map(pd.isna, m.iloc[:-1]))
    assert pd.isna(m.iloc[-1])


def test_isna(money_list):
    m = pd.Series(money_list, dtype="Money64").reindex(range(len(money_list) + 1))
    nas = m.isna()
    assert not any(nas.iloc[:-1])
    assert nas.iloc[-1]


def test_na_astype(money_list):
    m = (
        pd.Series(money_list, dtype="Money64")
        .reindex(range(len(money_list) + 1))
        .astype("Float64")
    )
    nas = m.isna()
    assert not any(nas.iloc[:-1])
    assert nas.iloc[-1]


def test_na_arithmetic(money_list):
    m1 = pd.Series(money_list, dtype="Money64").reindex(range(len(money_list) + 1))
    m2 = m1.iloc[::-1].reset_index(drop=True)
    for oper, func in [
        ("+", lambda x, y: x + y),
        ("-", lambda x, y: x - y),
        ("*", lambda x, y: x * y),
        ("/", lambda x, y: x / y),
    ]:
        if oper == "*":
            # builtins.TypeError: Cannot multiply money by money <MoneyArray>
            nas = func(m1, m2.astype("Float64")).isna()
        else:
            nas = func(m1, m2).isna()
        assert nas.iloc[0]
        assert nas.iloc[-1]
        assert not any(nas.iloc[1:-1])


def test_copy(money_list):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = m1.copy()
    assert all(m1 == m2)


def test_concat(money_list):
    m1 = pd.Series(money_list, dtype="Money64")
    m2 = m1.copy()

    m = pd.concat([m1, m2])
    for item, expect in zip(m, money_list + money_list):
        assert item == expect


def test_print_with_na(money_list):
    f = io.StringIO()
    m = pd.Series(money_list, dtype="Money64").reindex(range(len(money_list) + 1))

    with contextlib.redirect_stdout(f):
        print(m)

    lines = f.getvalue().splitlines()
    # adjust for slightly different formatting in GH Actions container
    assert re.match(r"\d\s+\$\s*3.14\s*", lines[0])
    assert re.match(r"\d\s+\$\s*2.72\s*", lines[1])
    assert re.match(r"\d\s+\<NA\>\s*", lines[2])
    assert "dtype: Money64" in lines[3]
