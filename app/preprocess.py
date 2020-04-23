import numpy as np
import typing as t
import pandas as pd
import re
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor
from skimage import io
import glob
from tqdm import tqdm
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold as _KFold
from .entities import Audios, Audio
from cytoolz.curried import unique, pipe, map, mapcat, frequencies, topk
import matplotlib.pyplot as plt
from app.config import NOISED_TGT_DIR, RAW_TGT_DIR
from librosa import display
from librosa.feature.inverse import mel_to_audio
from librosa.output import write_wav
import librosa


def load_audios(dirctory: str) -> Audios:
    paths = glob.glob(os.path.join(dirctory, "*.npy"))
    audios: Audios = []
    for p in paths:
        matched = re.search(r"\d+", p)
        if matched is not None:
            id = matched.group(0)
            spectrogram = np.load(p)
            audios.append(Audio(id, librosa.power_to_db(spectrogram)))
    return audios


def save_wav(spectrogram: t.Any, path: t.Union[str, Path]) -> None:
    signal = mel_to_audio(spectrogram)
    write_wav(path, signal, sr=22050)


def summary(audios: Audios) -> t.Dict[str, t.Any]:
    shapes = [x.spectrogram.shape for x in audios]
    spectrograms = [x.spectrogram for x in audios]
    length_range = (
        min([x[1] for x in shapes]),
        max([x[1] for x in shapes]),
    )

    value_range = (
        min([np.min(x) for x in spectrograms]),
        max([np.max(x) for x in spectrograms]),
    )
    return {
        "length_range": length_range,
        "value_range": value_range,
    }


def plot_spectrograms(
    spectrograms: t.Sequence[t.Any], path: t.Union[str, Path]
) -> None:
    fig, axs = plt.subplots(len(spectrograms), sharex=True)
    for sp, ax in zip(spectrograms, axs):
        im = ax.imshow(sp, interpolation="nearest", cmap="gray")
        fig.colorbar(im, ax=ax)
        ax.set_aspect("auto")
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def show_detail(spectrogram: t.Any, path: t.Union[str, Path]) -> None:
    fig, axs = plt.subplots(3, sharex=True)
    im = axs[0].imshow(spectrogram, interpolation="nearest", cmap="gray")
    axs[0].set_aspect("auto")
    diff = np.diff(spectrogram, axis=0)
    axs[1].imshow(diff, interpolation="nearest", cmap="gray")
    axs[1].set_aspect("auto")
    axs[2].plot(np.sum(spectrogram, axis=0))

    fig.tight_layout()
    plt.savefig(path)
    plt.close()


class KFold:
    def __init__(self, n_split: int = 5) -> None:
        self.n_split = n_split

    def __call__(self, audios: Audios) -> t.Iterator[t.Tuple[Audios, Audios]]:
        kf = _KFold(self.n_split, shuffle=False)
        for train, valid in kf.split(audios):
            yield [audios[i] for i in train], [audios[i] for i in valid]


class Noise:
    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, spectrogram: t.Any) -> t.Any:
        fill_value = np.random.choice(np.sort(spectrogram,axis=None))
        mask = np.random.choice(
            [True, False], size=spectrogram.shape, p=[self.p, (1 - self.p)]
        )
        masked = spectrogram + mask * fill_value
        return masked


class RandomCrop1d:
    def __init__(self, length: int) -> None:
        self.length = length

    def __call__(self, x: t.Any, y: t.Any) -> t.Tuple[t.Any, t.Any]:
        shape = x.shape
        high = shape[1] - self.length
        start = np.random.randint(low=0, high=high)
        return x[:, start : start + self.length], y[:, start : start + self.length]


class Flip1d:
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, x: t.Any, y: t.Any) -> t.Tuple[t.Any, t.Any]:
        is_enable = np.random.choice(
            [False, True], size=(1,), p=[self.p, (1 - self.p)]
        )[0]
        if is_enable:
            return x[:, ::-1].copy(), y[:, ::-1].copy()
        else:
            return x, y



class Scaler:
    def __init__(self, high: float, low: float) -> None:
        self.high = high
        self.low = low
        self._avg = (high + low) / 2

    def __call__(self, x: t.Any) -> t.Any:
        return x - self._avg
