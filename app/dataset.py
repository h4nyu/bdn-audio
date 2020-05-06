import numpy as np
import typing as t
from torch.utils.data import Dataset as _Dataset
from .entities import Audios, Audio
from .config import MAX_POWER
from .preprocess import (
    Noise,
    RandomCrop1d,
    Scaler,
    HFlip1d,
    VFlip1d,
    RandomScale,
    RandomCrop2d,
)
from .config import NOISE_P, NOISE_HIGH, NOISE_LOW
from sklearn.preprocessing import MinMaxScaler
from albumentations.augmentations.transforms import (
    Resize,
    RandomCrop,
    RandomGridShuffle,
    ElasticTransform,
    Blur,
    GaussianBlur,
    GridDistortion,
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
        noised = Noise(p=NOISE_P, high=NOISE_HIGH, low=NOISE_LOW)(raw.copy())
        _max = np.max(raw)
        scale = MAX_POWER
        raw = (raw) / scale
        noised = (noised) / scale
        if self.mode == "train":
            resized = RandomCrop(height=self.resolution[0], width=self.resolution[1])(
                image=noised, mask=raw
            )
            noised, raw = resized["image"], resized["mask"]
            noised, raw = HFlip1d(p=0.5)(noised, raw)
            #  noised, raw = VFlip1d(p=0.5)(noised, raw)
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
        scale = MAX_POWER
        sp = (sp) / scale
        hfliped, _ = HFlip1d(p=1)(sp, sp)
        #  vfliped, _ = VFlip1d(p=1)(sp, sp)
        return sp, hfliped, row.id, scale
