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


def show_detail(audio: Audio, path: t.Union[str, Path]) -> None:
    spectrogram = librosa.power_to_db(audio.spectrogram)
    fig, axs = plt.subplots(3, sharex=True)
    im = axs[0].imshow(spectrogram,interpolation='nearest',cmap='jet')
    axs[0].set_aspect('auto')
    axs[1].imshow(np.diff(spectrogram, axis=0),interpolation='nearest',cmap='jet')
    axs[1].set_aspect('auto')
    axs[2].plot(np.sum(spectrogram, axis=0))

    fig.tight_layout()
    plt.savefig(path)
    plt.close()

