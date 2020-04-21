import pytest
import librosa
from app.dataset import Dataset, Mode
from app.entities import Audios
from app.config import NOISED_TGT_DIR, RAW_TGT_DIR, CROP_LEN
from app.preprocess import load_audios, save_wav


def test_dataset() -> None:
    length = 32
    raw_audios = load_audios(RAW_TGT_DIR)
    dataset = Dataset(raw_audios, length)
    for i in range(2):
        x, y = dataset[0]
        x, y = librosa.db_to_power(x), librosa.db_to_power(y)
        save_wav(x, f"/store/wav/test-x-{i}.wav")
        save_wav(y, f"/store/wav/test-y-{i}.wav")
