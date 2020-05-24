import torch
from app.models import UNet2d


def test_unet2d() -> None:
    x = torch.randn(32, 128, 32)
    model = UNet2d()
    y = model(x)
    assert x.shape == y.shape
