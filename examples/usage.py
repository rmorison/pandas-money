from decimal import Decimal

import pandas as pd

import pandas_money as pm

s = pd.Series(["1.11", "2.22", "3.33"], dtype="Money64")

print(s)
print(s + s)
print(s * Decimal("1.5"))
print(s / 2)
print(s.iloc[0], type(s.iloc[0]))
