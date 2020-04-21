import numpy as np
import typing as t
from torch.utils.data import Dataset as _Dataset
from .entities import Audios, Audio
from .preprocess import Noise, RandomCrop1d
import librosa

Mode = t.Literal["Test", "Train"]


class Dataset(_Dataset):
    def __init__(
        self, audios: Audios, length: int, mode: t.Literal["train", "test"] = "train",
    ) -> None:
        self.audios = audios
        self.mode = mode
        self.length = length

    def __len__(self) -> int:
        return len(self.audios)

    def transform(self, audio: Audio) -> t.Tuple[t.Any, t.Any]:
        raw = audio.spectrogram.copy()
        noised = Noise(p=0.5)(audio.spectrogram.copy())
        noised, raw = RandomCrop1d(self.length)(noised, raw)
        noised = librosa.power_to_db(noised)
        raw = librosa.power_to_db(raw)
        return noised, raw

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, t.Any]:
        row = self.audios[idx]
        noised, raw = self.transform(row)
        return (noised, raw)
