from decimal import Decimal

import pandas as pd

import pandas_money as pm

s = pd.Series(["1.11", "2.22", "3.33"], dtype="money64")

print(s)
print(s + s)
print(s * Decimal("1.5"))
print(s / 2)
