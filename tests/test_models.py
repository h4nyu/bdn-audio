import torch
from app.models import SSE1d, CSE1d, SCSE1d, SENextBottleneck1d, Down


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


def test_senext_bottoleneck1d() -> None:
    x = torch.randn(32, 128, 32)
    model = SENextBottleneck1d(128, 64, is_shortcut=True)
    y = model(x)
    assert (x.shape[0], 64, x.shape[2]) == y.shape


def test_down() -> None:
    x = torch.randn(32, 128, 32)
    model = Down(128, 64)
    y = model(x)
    print(y.shape)
    assert (x.shape[0], 64, x.shape[2] // 2) == y.shape
