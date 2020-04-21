import torch
from app.models import SSE1d, CSE1d, SCSE1d


def test_sse1d() -> None:
    x = torch.randn(32, 128, 32)
    model = SSE1d(128)
    y = model(x)
    assert x.shape == y.shape


def test_cse1d() -> None:
    x = torch.randn(32, 128, 32)
    model = CSE1d(128, 16)
    y = model(x)
    assert x.shape == y.shape


def test_scse1d() -> None:
    x = torch.randn(32, 128, 32)
    model = SCSE1d(128, 16)
    y = model(x)
    assert x.shape == y.shape
