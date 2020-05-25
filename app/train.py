import numpy as np
import typing as t
import json
import os
from .entities import Audios, Audio
from .dataset import Dataset, PredictDataset
from .preprocess import HFlip1d, VFlip1d, Vote
import os
import torch
from pathlib import Path
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import mean_squared_error

from concurrent import futures
from datetime import datetime
from .models import UNet2d as NNModel
from torch.nn import MSELoss
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
    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        output_dir: Path,
        lr: float = 1e-2,
        check_interval: int = 10,
    ) -> None:
        self.device = DEVICE
        self.model = NNModel().to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)  # type: ignore
        self.epoch = 1
        self.data_loaders: DataLoaders = {
            "train": DataLoader(
                train_dataset, shuffle=True, batch_size=32, drop_last=True,
            ),
            "test": DataLoader(test_dataset, shuffle=True, batch_size=1,),
        }
        self.best_score = np.inf
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_path = self.output_dir.joinpath("checkpoint.json")
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, verbose=True, factor=0.5, eps=lr * 1e-2
        )

        if self.checkpoint_path.exists():
            self.load_checkpoint()

    def train_one_epoch(self) -> float:
        self.model.train()
        epoch_loss = 0.0
        score = 0.0
        count = 0
        base_score = 0.0
        for img, label, in tqdm(self.data_loaders["train"]):
            count = count + 1
            img, label = img.to(self.device).float(), label.to(self.device).float()
            pred = self.model(img)
            loss = self.objective(pred, label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()
            x = img[0].detach().cpu().numpy()
            pred = np.abs(pred[0].detach().cpu().numpy())
            y = label[0].detach().cpu().numpy()

        plot_spectrograms(
            [np.log(i) for i in [x, pred, y]], self.output_dir.joinpath(f"train.png"),
        )

        epoch_loss = epoch_loss / count
        epoch = self.epoch
        score = score / count
        return epoch_loss

    def objective(self, x: t.Any, y: t.Any) -> t.Any:
        #  loss = MSELoss()(x, y).mean() + MSELoss()(y.mean(), x.mean()).mean()
        loss = MSELoss()(x, y).mean()
        return loss

    def eval_one_epoch(self) -> t.Tuple[float, float, float]:
        self.model.eval()
        mean_vote = Vote("mean")
        epoch = self.epoch
        epoch_loss = 0.0
        score = 0.0
        base_score = 0.0
        count = 0
        for img, label, in tqdm(self.data_loaders["test"]):
            img, label = img.to(self.device).float(), label.to(self.device).float()
            count += 1
            with torch.no_grad():
                pred = self.model(img)
                loss = self.objective(pred, label)
                pred = self.model(img)[0].cpu().numpy()
                h_pred = self.model(img.flip(1)).flip(1)[0].cpu().numpy()
                v_pred = self.model(img.flip(2)).flip(2)[0].cpu().numpy()
                preds = [pred, h_pred, v_pred]
                pred = mean_vote(preds)
                epoch_loss += loss.item()
                x, y = [np.abs(i[0].detach().cpu().numpy()) for i in [img, label]]
                score += mean_squared_error(pred, y,)
                base_score += mean_squared_error(x, y,)

        plot_spectrograms(
            [np.log(i) for i in [x, pred, y]], self.output_dir.joinpath(f"eval.png"),
        )
        epoch_loss = epoch_loss / count
        score = score / count
        base_score = base_score / count
        return epoch_loss, base_score, score

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
        for epoch in range(max_epochs + 1):
            self.epoch = epoch
            train_loss = self.train_one_epoch()
            eval_loss, base_score, score = self.eval_one_epoch()
            logger.info(
                f"{epoch=} {train_loss=:.4f} {eval_loss=:.4f} {base_score=:.4f} {score=:.4f}"
            )
            self.scheduler.step(train_loss)
            if score < self.best_score:
                logger.info("update model")
                self.best_score = score
                self.save_checkpoint()


class Predict:
    def __init__(self, model_path: str, audios: Audios, output_dir: str) -> None:
        self.model = NNModel().double().to(DEVICE)
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
        mean_vote = Vote("mean")
        with torch.no_grad():
            for x, ids in self.data_loader:
                id = ids[0]
                x = x.to(DEVICE)
                y = self.model(x)[0].cpu().numpy()
                h_y = self.model(x.flip(1)).flip(1)[0].cpu().numpy()
                v_y = self.model(x.flip(2)).flip(2)[0].cpu().numpy()
                ys = [y, h_y, v_y]
                y = mean_vote(ys)
                predict_audios.append(Audio(id, y))
        return predict_audios
