import numpy as np
import typing as t
from torch.utils.data import Dataset as _Dataset
from .entities import Audios, Audio
from .preprocess import Noise, RandomCrop1d, Scaler, HFlip1d, VFlip1d, RandomScale, RandomCrop2d
from .config import VALUE_RANGE
from sklearn.preprocessing import MinMaxScaler
from albumentations.augmentations.transforms import Resize, RandomCrop, RandomGridShuffle, ElasticTransform
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
        noised = Noise(p=0.2, high=0.1, low=0.01)(raw.copy())
        raw = np.log(raw)
        noised = np.log(noised)
        _min = np.min(raw)
        _max = np.max(raw)
        scale = _max - _min
        raw = (raw - _min) / scale
        noised = (noised - _min) / scale
        if self.mode == "train":
            resized = RandomCrop(height=self.resolution[0], width=self.resolution[1])(
                image=noised, mask=raw
            )
            noised, raw = resized["image"], resized["mask"]
            noised, raw = HFlip1d(p=0.5)(noised, raw)
            noised, raw = RandomScale(p=1, high=1.01, low=0.1)(noised, raw)
        return noised, raw, scale, _min, _max

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
        sp = np.log(sp)
        _min = np.min(sp)
        _max = np.max(sp)
        scale = _max - _min
        sp = (sp - _min) / scale
        hfloped,  _ = HFlip1d(p=1)(sp, sp)
        return sp, hfloped, row.id, scale, _min
