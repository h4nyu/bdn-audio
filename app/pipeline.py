from pathlib import Path
import typing as t
from .cache import Cache
from .config import NOISED_TGT_DIR, RAW_TGT_DIR
from .entities import Audios, Audio
from .preprocess import load_audios, show_detail, save_wav, Noise, summary, KFold
from .train import Trainer, Predict
from concurrent import futures
from pathlib import Path
from logging import getLogger
import librosa
import numpy as np

logger = getLogger(__name__)
cache = Cache("/store/tmp")
plot_dir = Path("/store/plot")
plot_dir.mkdir(exist_ok=True)
wav_dir = Path("/store/wav")
wav_dir.mkdir(exist_ok=True)


def summary(in_path:str, out_path:str) -> t.Any:
    executor = futures.ProcessPoolExecutor()
    audios = load_audios(in_path)
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    futures.wait(
        [
            executor.submit(
                show_detail,
                audio.spectrogram,
                out_path.joinpath(f"{audio.id}.png"),
            )
            for audio in audios
        ]
    )

def eda() -> t.Any:
    executor = futures.ProcessPoolExecutor()
    noised_audios = load_audios(NOISED_TGT_DIR)

    futures.wait(
        [
            executor.submit(
                show_detail,
                audio.spectrogram,
                plot_dir.joinpath(f"noised-specgram-{audio.id}.png"),
            )
            for audio in noised_audios
        ]
    )

    futures.wait(
        [
            executor.submit(
                show_detail,
                audio.spectrogram,
                plot_dir.joinpath(f"noised-specgram-{audio.id}.png"),
            )
            for audio in noised_audios
        ]
    )
    futures.wait(
        [
            executor.submit(
                cache(f"noised-wav-{audio.id}", save_wav),
                audio,
                wav_dir.joinpath(f"noised-{audio.id}.wav"),
            )
            for audio in noised_audios
        ]
    )

    raw_audios = load_audios(RAW_TGT_DIR)
    futures.wait(
        [
            executor.submit(
                show_detail,
                audio.spectrogram,
                plot_dir.joinpath(f"raw-specgram-{audio.id}.png"),
            )
            for audio in raw_audios
        ]
    )
    futures.wait(
        [
            executor.submit(
                cache(f"raw-wav-{audio.id}", save_wav),
                audio,
                wav_dir.joinpath(f"raw-{audio.id}.wav"),
            )
            for audio in raw_audios
        ]
    )

    all_audios = noised_audios + raw_audios
    dataset_summary = summary(all_audios)
    logger.info(f"{dataset_summary=}")


def dummy_aug(p: float = 0.2) -> t.Any:
    executor = futures.ProcessPoolExecutor()
    raw_audios = load_audios(RAW_TGT_DIR)
    noise = Noise(p=p)
    futs, _ = futures.wait(
        [executor.submit(noise, audio.spectrogram) for audio in raw_audios]
    )
    noised_audios = [Audio(r.id, ft.result()) for r, ft in zip(raw_audios, futs)]
    futures.wait(
        [
            executor.submit(
                show_detail,
                audio.spectrogram,
                plot_dir.joinpath(f"generated-noised-specgram-p-{p}-{audio.id}.png"),
            )
            for audio in noised_audios
        ]
    )

    futures.wait(
        [
            executor.submit(
                save_wav, audio, wav_dir.joinpath(f"generated-raw-{audio.id}.wav"),
            )
            for audio in noised_audios
        ]
    )


def train() -> None:
    raw_audios = load_audios(RAW_TGT_DIR)
    kf = KFold(n_split=5)
    for i, (train, valid) in enumerate(kf(raw_audios)):
        t = Trainer(train, valid, output_dir=Path(f"/store/model-{i}"))
        t.train(4000)


def predict() -> None:
    noised_audios = load_audios(NOISED_TGT_DIR)
    clean_audios = Predict("/store/model-1/model.pth",noised_audios, "/store/predict-0")()
    power_spectrograms = [
        librosa.db_to_power(x.spectrogram)
        for x in clean_audios
    ]
    submit_dir = Path("/store/predict-0/submit")
    submit_dir.mkdir(exist_ok=True)
    for sp, audio in zip(power_spectrograms, clean_audios):
        sp = np.zeros_like(sp)
        np.save(submit_dir.joinpath(f"tgt_{audio.id}.npy"), sp)
