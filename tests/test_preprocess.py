import numpy as np
from concurrent import futures
from app.preprocess import load_audios, Noise, Merge, plot_spectrograms
from app.config import NOISED_TGT_DIR


def test_load_audios() -> None:
    res = load_audios(NOISED_TGT_DIR)
    assert len(res) == 30
    first_item = res[0]


def test_merge() -> None:
    audios = load_audios(NOISED_TGT_DIR)
    m = Merge(0.01)
    sp = audios[0].spectrogram
    res = m(sp, np.ones(sp.shape) * 0.0001)
    print(sp.shape)
    print(res.shape)
    plot_spectrograms([np.log(res), np.log(sp)], "/store/plot/merge.png")
