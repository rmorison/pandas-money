from decimal import Decimal

import numpy as np
import pandas as pd
from moneyed import USD, Money
from pandas._typing import Dtype
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
    register_extension_dtype,
)
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import is_dtype_equal, pandas_dtype
from pandas.core.dtypes.dtypes import PandasExtensionDtype

money_decimals = 2
money_factor = 10**money_decimals
decimal_quantize = Decimal(f".{'0'*(money_decimals-1)}1")

InputScalar = int | Money | Decimal | str | float | np.integer | np.floating


@register_extension_dtype
class MoneyDtype(PandasExtensionDtype):
    """DType class for MoneyArray extension"""

    type = Money
    name = "money64"

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype
        """
        return MoneyArray


class MoneyArray(ExtensionScalarOpsMixin, ExtensionArray):
    """A Pandas ExtensionArray of int64 for fixed decimal currency math

    Operator type rules

    MoneyArray ± (MoneyArray | Money) = MoneyArray
    MoneyArray * (int | float | Decimal | Iterable[int | float | Decimal]) = MoneyArray
    MoneyArray / (int | float | Decimal | Iterable[int | float | Decimal]) = MoneyArray
    MoneyArray / (MoneyArray | Money) = FloatArray
    """

    def __init__(self, values, copy=False, to_cents=True):
        self.values = np.array(
            [self._clean_value(v, to_cents) for v in values], dtype=np.int64
        )

    def _clean_value(self, v: InputScalar, to_cents: bool) -> int | Decimal:
        factor = money_factor if to_cents else 1
        round_to = money_decimals if to_cents else 0

        cleaned: int | np.integer | Decimal
        if isinstance(v, (int, np.integer)):
            cleaned = v
        elif isinstance(v, Money):
            cleaned = v.amount
        elif isinstance(v, Decimal):
            cleaned = round(v, round_to)
        elif isinstance(v, str):
            cleaned = round(Decimal(v), round_to)
        elif isinstance(v, (float, np.floating)):
            cleaned = round(Decimal(float(v)), round_to)
        else:
            raise TypeError(f"{v} cannot be converted to a Money type")
        return factor * cleaned

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        return MoneyArray(scalars, copy=copy)

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    def __getitem__(self, item):
        if isinstance(item, int):
            return Money(self.values[item], USD) / money_factor
        else:
            return MoneyArray(self.values[item], to_cents=False)

    def __eq__(self, other):
        if isinstance(other, (pd.Index, pd.Series, pd.DataFrame)):
            return NotImplemented
        if not isinstance(other, MoneyArray):
            other = MoneyArray(other)
        return self.values == other.values

    @property
    def dtype(self):
        return MoneyDtype()

    @property
    def nbytes(self):
        return self.values.nbytes

    def __len__(self):
        return self.values.size

    def isna(self):
        return np.isnan(self.values)

    def take(self, indices, *, allow_fill=False, fill_value=None):
        from pandas.core.algorithms import take

        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(
            self.values,
            indices,
            fill_value=fill_value,
            allow_fill=allow_fill,
        )
        return MoneyArray(result, to_cents=False)

    def copy(self):
        return MoneyArray(np.copy(self.values))

    @classmethod
    def _concat_same_type(cls, to_concat):
        return MoneyArray(np.concatenate(to_concat))

    def astype(self, dtype, copy: bool = True):
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(dtype, self.dtype):
            if not copy:
                return self
            else:
                return self.copy()

        decimals = [Decimal(int(v)) / money_factor for v in self.values]

        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            return cls._from_sequence(decimals, dtype=dtype, copy=False)  # type: ignore

        return np.array(decimals, dtype=dtype, copy=False)

    def __add__(self, other):
        if isinstance(other, Money):
            return MoneyArray(
                self.values + np.int64(other.amount * money_factor),
                to_cents=False,
            )
        elif isinstance(other, MoneyArray) and len(self) == len(other):
            return MoneyArray(self.values + other.values, to_cents=False)
        else:
            raise TypeError(f"Cannot add money to {other}")

    def __radd__(self, other):
        raise NotImplemented

    def __sub__(self, other):
        if isinstance(other, Money):
            return MoneyArray(
                self.values - np.int64(other.amount * money_factor),
                to_cents=False,
            )
        elif isinstance(other, MoneyArray) and len(self) == len(other):
            return MoneyArray(self.values - other.values, to_cents=False)
        else:
            raise TypeError(f"Cannot subtract money with {other}")

    def __rsub__(self, other):
        raise NotImplemented

    def __mul__(self, other):
        if isinstance(other, (Money, MoneyArray)):
            raise TypeError(f"Cannot multiply money by money {other}")
        elif isinstance(other, Decimal):
            return MoneyArray(self.values * float(other), to_cents=False)
        elif isinstance(other, (np.number, int, float)):
            return MoneyArray(self.values * other, to_cents=False)
        elif isinstance(other, (list, np.ndarray)) and len(self) == len(other):
            return MoneyArray(self.values * other, to_cents=False)
        else:
            raise TypeError(f"Cannot multiply money by {other}")

    def __rmul__(self, other):
        raise NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Money):
            return self.values / (np.float64(other.amount) * money_factor)
        elif isinstance(other, MoneyArray) and len(self) == len(other):
            return self.values / other.values.astype(np.float64)
        elif isinstance(other, Decimal):
            return MoneyArray(self.values / np.float64(float(other)), to_cents=False)
        elif isinstance(other, (np.number, int, float)):
            return MoneyArray(self.values / np.float64(other), to_cents=False)
        elif isinstance(other, (list, np.ndarray)) and len(self) == len(other):
            return MoneyArray(self.values / other, to_cents=False)
        else:
            raise TypeError(f"Cannot divide money by {other}")

    def __rtruediv__(self, other):
        raise NotImplemented


ops_overrides = [
    "eq",
    "add",
    "radd",
    "sub",
    "rsub",
    "mul",
    "rmul",
    "truediv",
    "rtruediv",
]
ops_overrides_patches = {
    f"__{op}__": getattr(MoneyArray, f"__{op}__") for op in ops_overrides
}
MoneyArray._add_arithmetic_ops()
MoneyArray._add_comparison_ops()
for attr, patch in ops_overrides_patches.items():
    setattr(MoneyArray, attr, patch)
