import torch
from app.models import (
    SSE1d,
    CSE1d,
    SCSE1d,
    SENextBottleneck1d,
    Down1d,
    Up1d,
    UNet1d,
    ResNext1d,
)


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
    model = Down1d(128, 64)
    y = model(x)
    assert (x.shape[0], 64, x.shape[2] // 2) == y.shape


def test_up() -> None:
    x0 = torch.randn(32, 128, 64)
    x1 = torch.randn(32, 64, 128)
    model = Up1d(128, 64)
    y = model(x0, x1)
    assert y.shape == (32, 64, 128)


def test_unet() -> None:
    x = torch.randn(32, 128, 32)
    model = UNet1d(128, 128)
    y = model(x)
    assert x.shape == y.shape


def test_res_next() -> None:
    x = torch.randn(32, 128, 32)
    model = ResNext1d(128, 128)
    y = model(x)
    assert x.shape == y.shape
