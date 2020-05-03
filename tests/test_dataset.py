import pytest
import librosa
import numpy as np
from app.dataset import Dataset, Mode, PredictDataset
from app.entities import Audios
from app.config import NOISED_TGT_DIR, RAW_TGT_DIR, CROP_LEN
from app.preprocess import load_audios, save_wav, plot_spectrograms


def test_dataset() -> None:
    raw_audios = load_audios(RAW_TGT_DIR)
    dataset = Dataset(raw_audios, (128, 128))
    for i in range(4):
        x, y, scale = dataset[0]
        x = x * scale
        y = y * scale
        plot_spectrograms([np.log(i) for i in [x, y]], f"/store/plot/test-{i}.png")
        #  save_wav(x, f"/store/wav/test-x-{i}.wav")
        #  save_wav(y, f"/store/wav/test-y-{i}.wav")


def test_predict_dataset() -> None:
    length = 32
    audios = load_audios(NOISED_TGT_DIR)
