import numpy as np
from concurrent import futures
from app.preprocess import load_audios, Noise, plot_spectrograms
from app.config import NOISED_TGT_DIR


def test_load_audios() -> None:
    res = load_audios(NOISED_TGT_DIR)
    assert len(res) == 30
    first_item = res[0]
