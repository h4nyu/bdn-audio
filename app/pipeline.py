from pathlib import Path
import typing as t
from .cache import Cache
from .config import NOISED_TGT_DIR, RAW_TGT_DIR
from .preprocess import load_audios, show_specgram, save_wav
from concurrent import futures
from pathlib import Path

cache = Cache("/srv/store/tmp")


def eda() -> t.Any:
    executor = futures.ProcessPoolExecutor()
    plot_dir = Path("/tmp/ccc")
    plot_dir.mkdir(exist_ok=True)
    noised_audios = load_audios(NOISED_TGT_DIR)
    plot_dir = Path("/srv/store/plot")
    plot_dir.mkdir(exist_ok=True)
    wav_dir = Path("/srv/store/wav")
    wav_dir.mkdir(exist_ok=True)

    futures.wait(
        [
            executor.submit(
                cache(f"noised-specgram-{audio.id}", show_specgram),
                audio,
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
                cache(f"raw-specgram-{audio.id}", show_specgram),
                audio,
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


def train() -> t.Any:
    ...
