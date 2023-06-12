# Pandas Money
A Pandas ArrayExtension for handling currency with int64 performance

## Getting started

### Add to your project
```
poetry add pandas-money
```
or with pip/pipenv
```
pipenv install pandas-money
```

### Example
```py
from decimal import Decimal

import pandas as pd

import pandas_money as pm

s = pd.Series(["1.11", "2.22", "3.33"], dtype="Money64")

print(s)
print(s + s)
print(s * Decimal("1.5"))
print(s / 2)
print(s.iloc[0], type(s.iloc[0]))
```
outputs
```
0    $1.11
1    $2.22
2    $3.33
dtype: Money64
0    $2.22
1    $4.44
2    $6.66
dtype: Money64
0    $1.66
1    $3.33
2    $5.00
dtype: Money64
0    $0.56
1    $1.11
2    $1.66
dtype: Money64
$1.11 <class 'moneyed.classes.Money'>
```

## Setup
- Clone repo
- `poetry install`

## Run tests

```shell
make test
```

## Format and lint before commit

```shell
make pre-commit
```
