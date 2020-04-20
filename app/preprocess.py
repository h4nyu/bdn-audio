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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
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
            audios.append(Audio(id, spectrogram))
    return audios


def save_wav(audio: Audio, path: t.Union[str, Path]) -> None:
    signal = mel_to_audio(audio.spectrogram)
    write_wav(path, signal, sr=22050)


def summary(audios: Audios) -> t.Dict[str, t.Any]:
    shapes = [x.spectrogram.shape for x in audios]
    length_range = (
        min([x[1] for x in shapes]),
        max([x[1] for x in shapes]),
    )
    return {"length_range": length_range}


def show_detail(audio: Audio, path: t.Union[str, Path]) -> None:
    spectrogram = librosa.power_to_db(audio.spectrogram)
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


class Noise:
    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, spectrogram: t.Any) -> t.Any:
        fill_value = np.min(spectrogram)
        mask = np.random.choice(
            [False, True], size=spectrogram.shape, p=[self.p, (1 - self.p)]
        )
        inverted_mask = np.logical_not(mask)
        masked = spectrogram * mask + inverted_mask * fill_value
        return masked


class RandomCrop1d:
    def __init__(self, length: int) -> None:
        self.length = length

    def __call__(self, x: t.Any, y: t.Any) -> t.Tuple[t.Any, t.Any]:
        shape = x.shape
        high = shape[1] - self.length
        start = np.random.randint(low=0, high=high)
        return x[:, start : start + self.length], y[:, start : start + self.length]


class ToDeciBell:
    def __init__(self,) -> None:
        ...
