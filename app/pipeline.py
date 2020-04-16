from pathlib import Path
import typing as t
from .cache import Cache

cache = Cache("/store/tmp")


def eda() -> t.Any:
    ...


def train() -> t.Any:
    ...
