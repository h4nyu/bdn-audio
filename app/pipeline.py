from pathlib import Path
import typing as t
from .dataset import Dataset, PredictDataset
from .cache import Cache
from .config import NOISED_TGT_DIR, RAW_TGT_DIR
from . import config
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
    Vote,
)
from .train import Trainer, Predict
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
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


def eda_noise() -> t.Any:
    clean_audios = load_audios(RAW_TGT_DIR)
    noise_audios = load_audios(NOISED_TGT_DIR)
    for index in range(0, 2):
        clean_sp = clean_audios[index].spectrogram[:, :128]
        noise_sp = noise_audios[index].spectrogram[:, :128]
        generated_noise_sp = Noise()(clean_sp)
        plot_spectrograms(
            [np.log(i) for i in [clean_sp, noise_sp]],
            plot_dir.joinpath(f"floor-{index}.png"),
        )
        clean_mean = clean_sp.mean()
        noised_mean = noise_sp.mean()
        generated_noise_mean = generated_noise_sp.mean()
        print(f"{clean_mean=} {noised_mean=} {generated_noise_mean=}")

        clean_std = clean_sp.std()
        noised_std = noise_sp.std()
        generated_noise_std = generated_noise_sp.std()
        print(f"{clean_std=} {noised_std=} {generated_noise_std=}")

        fig, axs = plt.subplots(3, sharex=True)
        bins = 100
        axs[0].hist(clean_sp.flatten(), bins=bins)
        axs[0].set_title("clean")

        axs[1].hist(noise_sp.flatten(), bins=bins)
        axs[1].set_title("noise")

        axs[2].hist(generated_noise_sp.flatten(), bins=bins)
        axs[2].set_title("generated_noise")
        plt.savefig(plot_dir.joinpath(f"floor_hist-{index}.png"))
        plt.close()


def eda_summary() -> None:
    audios = load_audios(NOISED_TGT_DIR)
    #  audios = load_audios(RAW_TGT_DIR)
    dataset_summary = summary(audios)
    logger.info(f"{dataset_summary=}")


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


def cross_section(in_path: str, out_path: str) -> t.Any:
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


def dummy_aug(p: float = 0.2) -> t.Any:
    executor = futures.ProcessPoolExecutor()
    raw_audios = load_audios(RAW_TGT_DIR)
    noise = Noise()
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


def train(fold_idx: int, lr: float, check_interval: int) -> None:
    raw_audios = load_audios(RAW_TGT_DIR)
    kf = KFold(n_split=4)
    train_data, test_data = list(kf(raw_audios))[fold_idx]
    resolution = (128, 32)
    train_dataset = Dataset(
        train_data * check_interval, resolution=resolution, mode="train",
    )
    test_dataset = Dataset(test_data, resolution=resolution, mode="test",)
    t = Trainer(
        train_dataset,
        test_dataset,
        output_dir=Path(f"/store/model-{fold_idx}"),
        lr=lr,
        check_interval=check_interval,
    )
    t.train(8000)


def pre_submit(indices: t.List[int]) -> None:
    raw_audios = load_audios(RAW_TGT_DIR)[:19]
    noise = Noise()
    noised_audios = [Audio(x.id, noise(x.spectrogram)) for x in raw_audios]

    submit_dir = Path("/store/pre_submit")
    submit_dir.mkdir(exist_ok=True)
    fold_preds = [
        Predict(f"/store/model-{i}/model.pth", noised_audios, str(submit_dir))()
        for i in indices
    ]
    score = 0
    base_score = 0.0
    count = 1
    length = 0
    for x, ys, gt in zip(noised_audios, zip(*fold_preds), raw_audios):
        count += 1
        x_sp = x.spectrogram
        y_spes = [i.spectrogram for i in ys]
        length += x_sp.shape[1]

        y_gt = gt.spectrogram
        merged = Vote("mean")(y_spes)
        print(np.max(merged), np.max(y_gt), np.max(x_sp))
        mse = Mse()
        score += mse(merged, y_gt)
        base_score += mse(x_sp, y_gt)
        plot_spectrograms(
            [np.log(x_sp), np.log(merged), np.log(y_gt),],
            submit_dir.joinpath(f"{x.id}.png"),
        )

        plot_spectrograms(
            [x_sp, merged, x_sp - merged,], submit_dir.joinpath(f"diff-{x.id}.png")
        )
    print(f"{score=} {base_score=} {length=}")


def submit(indices: t.List[int]) -> None:
    noised_audios = load_audios(NOISED_TGT_DIR)
    submit_dir = Path("/store/submit")
    submit_dir.mkdir(exist_ok=True)
    length = 0
    fold_preds = [
        Predict(f"/store/model-{i}/model.pth", noised_audios, str(submit_dir))()
        for i in indices
    ]
    for x, ys in zip(noised_audios, zip(*fold_preds)):
        x_sp = x.spectrogram
        length += x_sp.shape[1]
        y_spes = [y.spectrogram for y in ys]
        merged = np.max(np.stack(y_spes), axis=0)
        print(np.max(merged), np.max(x_sp))

        plot_spectrograms(
            [np.log(x_sp), np.log(merged),], submit_dir.joinpath(f"{x.id}.png")
        )

        plot_spectrograms(
            [x_sp, merged, x_sp - merged,], submit_dir.joinpath(f"diff-{x.id}.png")
        )

        np.save(file=submit_dir.joinpath(f"tgt_{x.id}.npy"), arr=merged)
    print(f"{length=}")
