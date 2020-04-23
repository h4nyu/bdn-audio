import pytest
import librosa
from app.dataset import Dataset, Mode
from app.entities import Audios
from app.config import NOISED_TGT_DIR, RAW_TGT_DIR, CROP_LEN
from app.preprocess import load_audios, save_wav, plot_spectrograms


def test_dataset() -> None:
    length = 32
    raw_audios = load_audios(RAW_TGT_DIR)
    dataset = Dataset(raw_audios, length)
    for i in range(4):
        x, y = dataset[0]
        plot_spectrograms([x, y, x-y], f"/store/plot/test-{i}.png")
        x, y = librosa.db_to_power(x), librosa.db_to_power(y)
        save_wav(x, f"/store/wav/test-x-{i}.wav")
        save_wav(y, f"/store/wav/test-y-{i}.wav")
