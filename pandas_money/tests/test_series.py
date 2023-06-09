from decimal import Decimal

import pandas as pd
import pytest
from moneyed import USD, Money

from pandas_money.series import (
    MoneySeries,
    money_quantize,
    money_type,
    ratio_decimals,
    ratio_digits,
    ratio_quantize,
    ratio_type,
)


@pytest.fixture
def data_dict():
    return {"a": Money("3.14", USD), "b": Money("2.72", USD)}


@pytest.fixture
def data_list(data_dict):
    return list(data_dict.values())


@pytest.fixture
def data_money():
    return Money("3.14", USD)


@pytest.fixture
def data_decimal():
    return Decimal("3.14")


@pytest.fixture
def data_str():
    return "3.14"


@pytest.fixture
def data_int():
    return 10


def test_data_dict(data_dict):
    m = MoneySeries(data_dict)
    assert m.dtype == money_type
    assert m["a"] == data_dict["a"].amount
    assert m["b"] == data_dict["b"].amount


def test_data_list(data_list):
    m = MoneySeries(data_list)
    assert m.dtype == money_type
    assert m.iloc[0] == data_list[0].amount
    assert m.iloc[1] == data_list[1].amount


def test_data_money(data_money):
    m = MoneySeries(data_money)
    assert m.dtype == money_type
    assert m.iloc[0] == data_money.amount


def test_data_decimal(data_decimal):
    m = MoneySeries(data_decimal)
    assert m.dtype == money_type
    assert m.iloc[0] == data_decimal


def test_data_str(data_str):
    m = MoneySeries(data_str)
    assert m.dtype == money_type
    assert m.iloc[0] == Decimal(data_str)


def test_data_int(data_int):
    m = MoneySeries(data_int)
    assert m.dtype == money_type
    assert m.iloc[0] == Decimal(data_int)


def test_rounding():
    round_down = MoneySeries(Decimal("1.444999"))
    round_up = MoneySeries(Decimal("1.445000"))
    assert round_down.iloc[0] == Decimal("1.44")
    assert round_up.iloc[0] == Decimal("1.45")


def test_add(data_list):
    m1 = MoneySeries(data_list)
    m2 = MoneySeries(data_list)
    m = m1 + m2
    assert m.dtype == money_type
    assert m.iloc[0] == data_list[0].amount + data_list[0].amount
    assert m.iloc[1] == data_list[1].amount + data_list[1].amount


def test_sub(data_list):
    m1 = MoneySeries(data_list)
    m2 = MoneySeries(data_list)
    m = m1 - m2
    assert m.dtype == money_type
    assert m.iloc[0] == Decimal(0)
    assert m.iloc[1] == Decimal(0)


def test_rsub(data_list):
    m1 = MoneySeries(data_list)
    m2 = pd.Series(data_list)
    with pytest.raises(TypeError):
        m2 - m1


def test_mul(data_list):
    m1 = MoneySeries(data_list)
    rate = Decimal(".1234")
    m = m1 * rate
    assert m.dtype == money_type
    assert m.iloc[0] == (data_list[0] * rate).amount.quantize(money_quantize)
    assert m.iloc[1] == (data_list[1] * rate).amount.quantize(money_quantize)


def test_rmul(data_list):
    m1 = MoneySeries(data_list)
    rate = Decimal(".1234")
    m = rate * m1
    assert m.dtype == money_type
    assert m.iloc[0] == (data_list[0] * rate).amount.quantize(money_quantize)
    assert m.iloc[1] == (data_list[1] * rate).amount.quantize(money_quantize)


def test_money_truediv(data_list):
    test_decimals = 4
    test_quantize = Decimal("1." + "0" * test_decimals)

    m1 = MoneySeries(data_list)
    m2 = MoneySeries(reversed(data_list))
    m = m1 / m2
    assert m.dtype == ratio_type
    assert m.iloc[0].quantize(test_quantize) == (
        data_list[0].amount / data_list[1].amount
    ).quantize(test_quantize)
    assert m.iloc[1].quantize(test_quantize) == (
        data_list[1].amount / data_list[0].amount
    ).quantize(test_quantize)


def test_truediv(data_list):
    test_decimals = 4
    test_quantize = Decimal("1." + "0" * test_decimals)

    m1 = MoneySeries(data_list)
    rate = Decimal(".1234")
    m = m1 / rate
    assert m.dtype == ratio_type
    assert m.iloc[0].quantize(test_quantize) == (data_list[0] / rate).amount.quantize(
        test_quantize
    )
    assert m.iloc[1].quantize(test_quantize) == (data_list[1] / rate).amount.quantize(
        test_quantize
    )


def test_rtruediv(data_list):
    m1 = MoneySeries(data_list)
    rate = Decimal(".1234")
    with pytest.raises(TypeError):
        m = rate / m1


def test_badscalar():
    with pytest.raises(TypeError):
        MoneySeries(1.1)


def test_badarray():
    with pytest.raises(TypeError):
        MoneySeries([1.1])
