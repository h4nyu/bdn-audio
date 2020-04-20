import pytest
from app.dataset import Dataset, Mode
from app.entities import Audios
from app.config import NOISED_TGT_DIR, RAW_TGT_DIR, CROP_LEN
from app.preprocess import load_audios


def test_dataset() -> None:
    raw_audios = load_audios(RAW_TGT_DIR)
    dataset = Dataset(raw_audios, CROP_LEN)
    for i in range(10):
        item, _ = dataset[0]
        item.shape[1] == CROP_LEN
