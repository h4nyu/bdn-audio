import numpy as np
import typing as t
import json
import os
from .entities import Audios
from .dataset import Dataset
import os
import torch
from pathlib import Path
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from mlboard_client import Writer
from sklearn.metrics import mean_squared_error
from concurrent import futures
from datetime import datetime
from .models import UNet
from logging import getLogger
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
        alpha = 1
        beta = 1
        gamma = 1
        coefficient = 3
        flops_multiplier = alpha * (beta ** 2) * (gamma ** 2)
        depth = 3 * alpha ** coefficient
        resolution = int(16 * beta ** coefficient)
        width = int(64 * gamma ** coefficient)
        logger.info(f"{alpha=}, {beta=}, {gamma=}, {flops_multiplier=}, {coefficient=}")
        logger.info(f"{resolution=}, {width=}, {depth=} ")
        self.model = UNet(in_channels=128, out_channels=128).double().to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters())  # type: ignore
        self.objective = nn.MSELoss()
        self.epoch = 1
        self.data_loaders: DataLoaders = {
            "train": DataLoader(
                Dataset(train_data, length=resolution, mode="train",),
                shuffle=True,
                batch_size=8,
            ),
            "test": DataLoader(
                Dataset(test_data, length=resolution, mode="test",),
                shuffle=True,
                batch_size=8,
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
        for img, label in tqdm(self.data_loaders["train"]):
            img, label = img.to(self.device), label.to(self.device)
            pred = self.model(img)
            loss = self.objective(pred, label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(self.data_loaders["train"])
        epoch = self.epoch
        logger.info(f"{epoch=} train {epoch_loss=}")

    def eval_one_epoch(self) -> t.Tuple[float, float]:
        self.model.eval()
        epoch = self.epoch
        epoch_loss = 0.0
        score = 0.0
        for img, label in tqdm(self.data_loaders["test"]):
            img, label = img.to(self.device), label.to(self.device)
            with torch.no_grad():
                pred = self.model(img)
                loss = self.objective(pred, label)
                epoch_loss += loss.item()
                score += loss.item()

        plot_spectrograms(
            [
                pred[0].cpu().numpy(),
                (label - img)[0].cpu().numpy(),
                img[0].cpu().numpy(),
                label[0].cpu().numpy(),
            ],
            self.output_dir.joinpath(f"eval-{self.epoch}.png"),
        )
        epoch_loss = epoch_loss / len(self.data_loaders["test"])
        score = score / len(self.data_loaders["test"])
        logger.info(f"{epoch=} test {epoch_loss=} {score=}")
        return epoch_loss, score

    def load_checkpoint(self,) -> None:
        with open(self.checkpoint_path, "r") as f:
            data = json.load(f)
            self.epoch = data["epoch"]
            self.best_score = data["best_score"]

        self.model.load_state_dict(
            torch.load(self.output_dir.joinpath(f"model.pth"))
        )

    def save_checkpoint(self,) -> None:
        with open(self.checkpoint_path, "w") as f:
            json.dump({"epoch": self.epoch, "best_score": self.best_score,}, f)
        torch.save(
            self.model.state_dict(), self.output_dir.joinpath(f"model.pth")
        )

    def train(self, max_epochs: int) -> None:
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch()
            if epoch % 10 == 0:
                _, score = self.eval_one_epoch()
                if score < self.best_score:
                    self.save_checkpoint()
                    self.best_score = score
