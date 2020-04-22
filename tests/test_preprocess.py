import numpy as np
from concurrent import futures
from app.preprocess import load_audios, Noise
from app.config import NOISED_TGT_DIR


def test_load_audios() -> None:
    res = load_audios(NOISED_TGT_DIR)
    assert len(res) == 30
    first_item = res[0]


def test_noise() -> None:
    arr = np.array(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    )
    Noise()(arr)
