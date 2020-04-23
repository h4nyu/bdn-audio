import numpy as np
import typing as t
from torch.utils.data import Dataset as _Dataset
from .entities import Audios, Audio
from .preprocess import Noise, RandomCrop1d, Scaler, Flip1d
from sklearn.preprocessing import MinMaxScaler
import librosa

Mode = t.Literal["Test", "Train"]


class Dataset(_Dataset):
    def __init__(
        self,
        audios: Audios,
        length: int,
        mode: t.Literal["train", "test"] = "train",
        high=30,
        low=-65,
    ) -> None:
        self.audios = audios
        self.mode = mode
        self.length = length
        self.high = 30
        self.low = -65

    def __len__(self) -> int:
        return len(self.audios)

    def transform(self, audio: Audio) -> t.Tuple[t.Any, t.Any]:
        raw = audio.spectrogram.copy()
        noised = Noise(p=0.01)(audio.spectrogram.copy())
        noised, raw = RandomCrop1d(self.length)(noised, raw)
        noised, raw = Flip1d(p=0.5)(noised, raw)
        return noised, raw

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, t.Any]:
        row = self.audios[idx]
        noised, raw = self.transform(row)
        return (noised, raw)
