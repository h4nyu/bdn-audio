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
import seaborn as sns
from app.config import NOISED_TGT_DIR, RAW_TGT_DIR
from librosa import display
from librosa.feature.inverse import mel_to_audio
from librosa.output import write_wav

sns.set()


def load_audios(dirctory: str) -> Audios:
    paths = glob.glob(os.path.join(dirctory, "*.npy"))
    audios: Audios = []
    for p in paths:
        matched = re.search(r"\d+", p)
        if matched is not None:
            id = matched.group(0)
            waveform = np.load(p)
            audios.append(Audio(id, waveform))
    return audios


def save_wav(audio: Audio, path: t.Union[str, Path]) -> None:
    signal = mel_to_audio(audio.waveform)
    write_wav(path, signal, sr=22050)


def show_specgram(audio: Audio, path: t.Union[str, Path]) -> None:
    # https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
    display.specshow(audio.waveform)
    plt.colorbar()
    plt.savefig(path)
    plt.close()
