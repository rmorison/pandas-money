from __future__ import annotations

from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
import pyarrow as pa  # type: ignore
from moneyed import Money

# Decimal constants
money_decimals = 2
money_digits = 16
ratio_decimals = 18
ratio_digits = 38
money_quantize = Decimal("1." + "0" * money_decimals)
ratio_quantize = Decimal("1." + "0" * ratio_decimals)
decimal_round = ROUND_HALF_UP

# Pyarrow constants
decimal_type_factory = pa.decimal128
money_type = pd.ArrowDtype(decimal_type_factory(money_digits, scale=money_decimals))
ratio_type = pd.ArrowDtype(decimal_type_factory(ratio_digits, scale=ratio_decimals))
float_type = pd.ArrowDtype(pa.float64())


class MoneySeries(pd.Series):
    decimals = money_decimals

    def __init__(
        self,
        data=None,
        **kwargs,
    ) -> None:
        if data is not None:
            if isinstance(data, dict):
                data = {k: self._marshal_item(v) for k, v in data.items()}
            elif isinstance(data, (int, str, Decimal, Money)):
                data = self._marshal_item(data)
            elif isinstance(data, Iterable):
                data = [self._marshal_item(v) for v in data]
            else:
                raise TypeError(
                    f"{type(data)} is not type int, str, or Decimal, or an iterable of those types"
                )
            kwargs["data"] = data
            kwargs["dtype"] = money_type
        super().__init__(
            **kwargs,
        )

    def _marshal_item(self, data):
        if isinstance(data, Money):
            data = data.amount
        elif isinstance(data, (str, int)):
            data = Decimal(data)
        elif isinstance(data, Decimal):
            ...
        else:
            raise TypeError(
                f"{data} is not a Money, Decimal, str or int type, required by MoneySeries"
            )
        return data.quantize(money_quantize, decimal_round)

    def __add__(self, other):
        return super().__add__(other).astype(money_type)

    def __sub__(self, other):
        return super().__sub__(other).astype(money_type)

    def __rsub__(self, other):
        raise TypeError("Cannot subtract non-Money by a Money instance.")

    def __mul__(self, other):
        return super().__mul__(other).round(money_decimals).astype(money_type)

    def __rmul__(self, other):
        return super().__rmul__(other).round(money_decimals).astype(money_type)

    def __truediv__(self, other):
        return super().__truediv__(other).round(ratio_decimals).astype(ratio_type)

    def __rtruediv__(self, other):
        raise TypeError("Cannot divide non-Money by a Money instance.")
