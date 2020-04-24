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
    ) -> None:
        self.audios = audios
        self.mode = mode
        self.length = length

    def __len__(self) -> int:
        return len(self.audios)

    def transform(self, audio: Audio) -> t.Tuple[t.Any, t.Any]:
        raw = audio.spectrogram.copy()
        noised = Noise(p=0.02)(audio.spectrogram.copy())
        if self.mode == "train":
            noised, raw = RandomCrop1d(self.length)(noised, raw)
            noised, raw = Flip1d(p=0.5)(noised, raw)
        return noised, raw

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, t.Any]:
        row = self.audios[idx]
        noised, raw = self.transform(row)
        return (noised, raw)

class PredictDataset(_Dataset):
    def __init__(
        self,
        audios: Audios,
    ) -> None:
        self.audios = audios

    def __len__(self) -> int:
        return len(self.audios)

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, str]:
        row = self.audios[idx]
        return row.spectrogram, row.id
