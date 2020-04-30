import numpy as np
import typing as t
from torch.utils.data import Dataset as _Dataset
from .entities import Audios, Audio
from .preprocess import Noise, RandomCrop1d, Scaler, Flip1d, RandomScale, RandomCrop2d
from .config import VALUE_RANGE
from sklearn.preprocessing import MinMaxScaler
from albumentations.augmentations.transforms import Resize, RandomCrop
import librosa

Mode = t.Literal["Test", "Train"]


class Dataset(_Dataset):
    def __init__(
        self,
        audios: Audios,
        resolution: int,
        mode: t.Literal["train", "test"] = "train",
    ) -> None:
        self.audios = audios
        self.mode = mode
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.audios)

    def transform(self, audio: Audio) -> t.Tuple[t.Any, t.Any]:
        raw = audio.spectrogram.copy()
        scale = np.max(raw)
        raw = raw / scale
        if self.mode == "train":
            shape = raw.shape
            low = (
                shape[1] * 0.8 if shape[1] * 0.8 > self.resolution else self.resolution
            )
            w = np.random.randint(low=low, high=shape[1] * 1.2)
            raw = Resize(height=128, width=w)(image=raw)["image"]
        noised = Noise(p=0.7, high=1, low=0.001)(raw.copy())
        if self.mode == "train":
            resized = RandomCrop(height=self.resolution, width=self.resolution)(
                image=noised, mask=raw
            )
            noised, raw = resized["image"], resized["mask"]
            noised, raw = Flip1d(p=0.5)(noised, raw)
        return noised, raw, scale

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, t.Any, t.Any]:
        row = self.audios[idx]
        noised, raw, scale = self.transform(row)
        return (noised, raw, scale)


class PredictDataset(_Dataset):
    def __init__(self, audios: Audios,) -> None:
        self.audios = audios

    def __len__(self) -> int:
        return len(self.audios)

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, str]:
        row = self.audios[idx]
        sp = row.spectrogram
        _min = np.min(sp)
        _max = np.max(sp)
        scale = _max
        sp = sp / scale
        hfloped,  _ = Flip1d(p=1)(sp, sp)
        return sp, hfloped, row.id, scale
