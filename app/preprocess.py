import numpy as np
import typing as t
import pandas as pd
import re
import os
from concurrent.futures import ProcessPoolExecutor
from skimage import io
import glob
from tqdm import tqdm
from sklearn.metrics import fbeta_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from .entities import Audios, Audio
from cytoolz.curried import unique, pipe, map, mapcat, frequencies, topk
import seaborn as sns
from app.config import NOISED_TGT_DIR

sns.set()


def load_audios() -> Audios:
    paths = glob.glob(os.path.join(NOISED_TGT_DIR, "*.npy"))
    audios: Audios = []
    for p in paths:
        matched = re.search(r"\d+", p)
        if matched is not None:
            id = matched.group(0)
            waveform = np.load(p)
            audios.append(Audio(id, waveform))
    return audios
