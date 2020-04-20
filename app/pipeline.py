from pathlib import Path
import typing as t
from .cache import Cache
from .config import NOISED_TGT_DIR, RAW_TGT_DIR
from .preprocess import load_audios, show_detail, save_wav, Noise, summary
from concurrent import futures
from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)
cache = Cache("/store/tmp")
plot_dir = Path("/store/plot")
plot_dir.mkdir(exist_ok=True)
wav_dir = Path("/store/wav")
wav_dir.mkdir(exist_ok=True)


def eda() -> t.Any:
    executor = futures.ProcessPoolExecutor()
    noised_audios = load_audios(NOISED_TGT_DIR)

    futures.wait(
        [
            executor.submit(
                show_detail,
                audio,
                plot_dir.joinpath(f"noised-specgram-{audio.id}.png"),
            )
            for audio in noised_audios
        ]
    )

    futures.wait(
        [
            executor.submit(
                show_detail,
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
                show_detail, audio, plot_dir.joinpath(f"raw-specgram-{audio.id}.png"),
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
    futs, _ = futures.wait([executor.submit(noise, audio,) for audio in raw_audios])
    noised_audios = [r.result() for r in futs]
    futures.wait(
        [
            executor.submit(
                show_detail,
                audio,
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


def train() -> t.Any:
    ...
