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
import pandas as pd
import pandas_money as pm
from decimal import Decimal

s = pd.Series(["1.11", "2.22", "3.33"], dtype="money64")

print(s)
print(s + s)
print(s * Decimal("1.5"))
print(s / 2)
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
