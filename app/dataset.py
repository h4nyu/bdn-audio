import numpy as np
from cytoolz.curried import unique, pipe, map, mapcat, frequencies, topk
import typing as t
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset as _Dataset
from .entities import Audios, Audio
from .preprocess import (
    Noise,
    HFlip1d,
    VFlip1d,
)
from albumentations.augmentations.transforms import RandomCrop
import librosa

Mode = t.Literal["Test", "Train"]


class Dataset(_Dataset):
    def __init__(
        self,
        audios: Audios,
        resolution: t.Tuple[int, int],
        mode: t.Literal["train", "test"] = "train",
    ) -> None:
        self.audios = audios
        self.mode = mode
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.audios)

    def transform(self, audio: Audio) -> t.Tuple[t.Any, t.Any]:
        raw = audio.spectrogram.copy()
        noised = Noise()(raw.copy())
        _max = np.max(noised)
        if self.mode == "train":
            _, w = raw.shape
            if w < self.resolution[1]:
                repeat = self.resolution[1] // w + 1
                noised = np.concatenate([noised] * repeat, axis=1)
                raw = np.concatenate([raw] * repeat, axis=1)

            resized = RandomCrop(height=self.resolution[0], width=self.resolution[1])(
                image=noised, mask=raw
            )
            noised, raw = resized["image"], resized["mask"]
            noised, raw = HFlip1d(p=0.5)(noised, raw)
            noised, raw = VFlip1d(p=0.5)(noised, raw)
        return noised, raw

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, t.Any]:
        row = self.audios[idx]
        return self.transform(row)


class PredictDataset(_Dataset):
    def __init__(self, audios: Audios,) -> None:
        self.audios = audios

    def __len__(self) -> int:
        return len(self.audios)

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, str]:
        row = self.audios[idx]
        sp = row.spectrogram
        _max = np.max(sp)
        return sp, row.id
