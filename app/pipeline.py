from pathlib import Path
import typing as t
from .cache import Cache
from .config import NOISED_TGT_DIR, RAW_TGT_DIR
from .entities import Audios, Audio
from .preprocess import (
    load_audios,
    show_detail,
    save_wav,
    Noise,
    summary,
    KFold,
    ToAudio,
    ToMel,
    Mse,
    plot_spectrograms,
)
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


def eda(in_path: str, out_path: str) -> t.Any:
    executor = futures.ProcessPoolExecutor()
    audios = load_audios(in_path)
    out_dir = Path(out_path)
    out_dir.mkdir(exist_ok=True)

    futures.wait(
        [
            executor.submit(
                show_detail, audio.spectrogram, out_dir.joinpath(f"{audio.id}.png"),
            )
            for audio in audios
        ]
    )
    futures.wait(
        [
            executor.submit(
                cache(f"{out_dir}-wav-{audio.id}", save_wav),
                audio,
                out_dir.joinpath(f"{audio.id}.wav"),
            )
            for audio in audios
        ]
    )
    dataset_summary = summary(audios)
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


def mel_to_audio() -> None:
    audios = load_audios(RAW_TGT_DIR)[:2]
    spectrograms = [x.spectrogram for x in audios]
    to_audio = ToAudio()
    to_mel = ToMel()
    timeseries = [to_audio(i) for i in spectrograms]

    inv_spectrograms = [to_mel(i) for i in timeseries]
    mse = Mse()
    a = sum([mse(x, y) for x, y in zip(inv_spectrograms, spectrograms)])
    print(a)


def train(fold_idx:int) -> None:
    raw_audios = load_audios(RAW_TGT_DIR)
    kf = KFold(n_split=10)
    train, valid = list(kf(raw_audios))[fold_idx]
    t = Trainer(train, valid, output_dir=Path(f"/store/model-{fold_idx}"))
    t.train(4000)


def predict() -> None:
    noised_audios = load_audios(NOISED_TGT_DIR)[:10]
    submit_dir = Path("/store/predict")
    submit_dir.mkdir(exist_ok=True)
    fold_preds = [
        Predict(
            f"/store/model-{i}/model.pth", noised_audios, submit_dir
        )()
        for i
        in [0]
    ]
    for x, ys in zip(noised_audios, zip(*fold_preds)):
        x_sp = x.spectrogram
        y_spes = [
            y.spectrogram
            for y
            in ys
        ]
        merged = sum(y_spes) / len(y_spes)
        #  print(np.isnan(merged).sum())
        #  adjust =  x_sp.max() / merged.max()
        adjust =  x_sp.mean() / merged.mean()
        merged = adjust * merged
        print(np.max(merged), np.max(x_sp))

        plot_spectrograms(
            [
                np.log(x_sp),
                np.log(merged),
            ],
            submit_dir.joinpath(f"{x.id}.png")
        )

        plot_spectrograms(
            [
                x_sp,
                merged,
                x_sp - merged,
            ],
            submit_dir.joinpath(f"diff-{x.id}.png")
        )


        np.save(file=submit_dir.joinpath(f"tgt_{x.id}.npy"), arr=merged)
