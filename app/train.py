import numpy as np
import typing as t
import json
import os
from .entities import Audios, Audio
from .dataset import Dataset, PredictDataset
from .config import VALUE_RANGE
from .preprocess import Flip1d
import os
import torch
from pathlib import Path
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error
from concurrent import futures
from datetime import datetime

#  from .models import UNet1d as NNModel
from .models import UNet2d as NNModel

#  from .models import LogCoshLoss as Loss
from torch.nn import L1Loss as Loss
from logging import getLogger
import librosa
from tqdm import tqdm
from torchvision.transforms import ToTensor
from .preprocess import plot_spectrograms

#
logger = getLogger(__name__)
DEVICE = torch.device("cuda")
DataLoaders = t.TypedDict("DataLoaders", {"train": DataLoader, "test": DataLoader,})


class Trainer:
    def __init__(self, train_data: Audios, test_data: Audios, output_dir: Path) -> None:
        self.device = DEVICE
        resolution = 32
        self.model = NNModel(in_channels=128, out_channels=128).double().to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters())  # type: ignore
        self.objective = Loss()
        self.epoch = 1
        self.data_loaders: DataLoaders = {
            "train": DataLoader(
                Dataset(train_data, resolution=resolution, mode="train",),
                shuffle=True,
                batch_size=16,
                drop_last=True,
            ),
            "test": DataLoader(
                Dataset(test_data, resolution=resolution, mode="test",),
                shuffle=True,
                batch_size=1,
            ),
        }
        self.best_score = np.inf
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_path = self.output_dir.joinpath("checkpoint.json")
        train_len = len(train_data)
        logger.info(f"{train_len=}")
        test_len = len(test_data)
        logger.info(f"{test_len=}")

        if self.checkpoint_path.exists():
            self.load_checkpoint()

    def train_one_epoch(self) -> None:
        self.model.train()
        epoch_loss = 0.0
        score = 0.0
        count = 0
        base_score = 0.0
        for img, label, _ in tqdm(self.data_loaders["train"]):
            count = count + 1
            img, label = img.to(self.device), label.to(self.device)
            pred = self.model(img)
            loss = self.objective(pred, label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()

        x = img[0].detach().cpu().numpy()
        pred = pred[0].detach().cpu().numpy()
        y = label[0].detach().cpu().numpy()
        score += mean_squared_error(pred, y,)
        base_score += mean_squared_error(x, y,)

        plot_spectrograms(
            [x, pred, y], self.output_dir.joinpath(f"train.png"),
        )

        epoch_loss = epoch_loss / count
        epoch = self.epoch
        score = score / count
        base_score = base_score / count
        score_diff = base_score - score
        logger.info(
            f"{epoch=} train {epoch_loss=} {base_score=} {score=} {score_diff=}"
        )

    def eval_one_epoch(self) -> t.Tuple[float, float]:
        self.model.eval()
        epoch = self.epoch
        epoch_loss = 0.0
        score = 0.0
        base_score = 0.0
        count = 0
        for img, label, _ in tqdm(self.data_loaders["test"]):
            img, label = img.to(self.device), label.to(self.device)
            count += 1
            with torch.no_grad():
                pred = self.model(img)
                loss = self.objective(pred, label)
                epoch_loss += loss.item()

                x = img[0].cpu().numpy()
                pred = pred[0].cpu().numpy()
                y = label[0].cpu().numpy()
                score += mean_squared_error(pred, y,)
                base_score += mean_squared_error(x, y,)

        plot_spectrograms(
            [x, pred, y], self.output_dir.joinpath(f"eval.png"),
        )
        epoch_loss = epoch_loss / count
        score = score / count
        base_score = base_score / count
        score_diff = base_score - score
        logger.info(f"{epoch=} test {epoch_loss=} {base_score=} {score=} {score_diff=}")
        return epoch_loss, score

    def load_checkpoint(self,) -> None:
        with open(self.checkpoint_path, "r") as f:
            data = json.load(f)
            self.epoch = data["epoch"]
            self.best_score = data["best_score"]

        self.model.load_state_dict(torch.load(self.output_dir.joinpath(f"model.pth")))

    def save_checkpoint(self,) -> None:
        with open(self.checkpoint_path, "w") as f:
            json.dump({"epoch": self.epoch, "best_score": self.best_score,}, f)
        torch.save(self.model.state_dict(), self.output_dir.joinpath(f"model.pth"))

    def train(self, max_epochs: int) -> None:
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch()
            _, score = self.eval_one_epoch()
            self.save_checkpoint()
            self.best_score = score


class Predict:
    def __init__(self, model_path: str, audios: Audios, output_dir: str) -> None:
        self.model = NNModel(in_channels=128, out_channels=128).double().to(DEVICE)
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.audios = audios
        self.length = 16
        self.data_loader = DataLoader(
            PredictDataset(audios), shuffle=False, batch_size=1,
        )

    def __call__(self,) -> Audios:
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        predict_audios: Audios = []
        flip = Flip1d(p=1)
        with torch.no_grad():
            for x, hfliped, ids, _mins, scales in self.data_loader:
                id = ids[0]
                _min = _mins[0].item()
                scale = scales[0].item()
                x, hfliped = x.to(DEVICE), hfliped.to(DEVICE)
                y = self.model(x)[0].cpu().numpy()
                h_y = self.model(hfliped)[0].cpu().numpy()
                y = (y + flip(h_y, h_y)[0]) / 2
                y = np.exp(y * scale + _min)
                predict_audios.append(Audio(id, y))
        return predict_audios
