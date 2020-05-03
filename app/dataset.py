import numpy as np
import typing as t
from torch.utils.data import Dataset as _Dataset
from .entities import Audios, Audio
from .preprocess import (
    Noise,
    RandomCrop1d,
    Scaler,
    HFlip1d,
    VFlip1d,
    RandomScale,
    RandomCrop2d,
)
from .config import VALUE_RANGE
from sklearn.preprocessing import MinMaxScaler
from albumentations.augmentations.transforms import (
    Resize,
    RandomCrop,
    RandomGridShuffle,
    ElasticTransform,
    Blur,
)
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
        noised = Noise(p=0.5, high=0.2, low=0.01)(raw.copy())
        _max = np.max(raw)
        scale = _max
        raw = (raw) / scale
        noised = (noised) / scale
        if self.mode == "train":
            resized = RandomCrop(height=self.resolution[0], width=self.resolution[1])(
                image=noised, mask=raw
            )
            noised = Blur(p=1, blur_limit=32)(image=noised)["image"]
            noised, raw = resized["image"], resized["mask"]
            noised, raw = HFlip1d(p=0.5)(noised, raw)
        return noised, raw, scale

    def __getitem__(self, idx: int) -> t.Tuple[t.Any, t.Any, t.Any]:
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
        scale = _max
        sp = (sp) / scale
        hfloped, _ = HFlip1d(p=1)(sp, sp)
        return sp, hfloped, row.id, scale
