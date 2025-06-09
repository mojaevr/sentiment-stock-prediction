# train_lstm_lightning.py
"""
Train LSTM model using PyTorch Lightning for
stock market prediction with sentiment feat.
"""
import os
import subprocess

import hydra
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_size, hidden_size=200, dropout=0.2, lr=1e-3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def create_dataset(dataset, back=1):
    X, Y = [], []
    for i in range(len(dataset) - back - 1):
        fin = i + back
        a = dataset[i:fin, 0]
        X.append(a)
        Y.append(dataset[fin, 0])
    return np.array(X), np.array(Y)


def get_dataloaders(X_train_all_feat, y_train, X_test_all_feat, y_test, bs=25):
    X_train = torch.tensor(X_train_all_feat, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test_all_feat, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
    train_ds = TensorDataset(X_train, y_train_t)
    test_ds = TensorDataset(X_test, y_test_t)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False)
    val_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # MLflow setup
    mlflow_cfg = hydra.compose(config_name="mlflow")
    mlflow.set_tracking_uri(mlflow_cfg.mlflow.tracking_uri)
    mlflow.set_experiment(mlflow_cfg.mlflow.experiment_name)
    os.makedirs(mlflow_cfg.mlflow.plots_dir, exist_ok=True)

    # Get git commit id
    try:
        commit_id = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit_id = "unknown"

    # Data loading and preprocessing
    result = pd.read_pickle(cfg.paths.dataset_path)

    # Target variable
    y = result.MSFT.values.astype("float32").reshape(-1, 1)
    scaler = MinMaxScaler(
        feature_range=(
            cfg.preprocessing.scaler.feature_range.min,
            cfg.preprocessing.scaler.feature_range.max,
        )
    )
    y = scaler.fit_transform(y)

    # Sentiment feature
    X = result.open.values.astype("float32").reshape(-1, 1)

    train_size = int(len(y) * cfg.preprocessing.percent_of_training)
    train_y, test_y = y[:train_size], y[train_size:]
    train_x, test_x = X[:train_size], X[train_size:]

    X_train_feat_1, y_train = create_dataset(train_y, cfg.preprocessing.back)
    X_test_feat_1, y_test = create_dataset(test_y, cfg.preprocessing.back)
    X_train_feat_2, _ = create_dataset(train_x, cfg.preprocessing.back)
    X_test_feat_2, _ = create_dataset(test_x, cfg.preprocessing.back)

    # Формируем shape (batch, seq_len=back, feat=2)
    X_train_all_feat = np.stack([X_train_feat_1, X_train_feat_2], axis=1)
    X_test_all_feat = np.stack([X_test_feat_1, X_test_feat_2], axis=1)

    train_loader, val_loader = get_dataloaders(
        X_train_all_feat,
        y_train,
        X_test_all_feat,
        y_test,
        bs=cfg.training.batch_size,
    )

    print("shape =", X_train_all_feat.shape)
    model = LSTMRegressor(
        input_size=X_train_all_feat.shape[2],
        # hidden_size=cfg.model.hidden_size,
        # dropout=cfg.model.dropout,
        # lr=cfg.model.lr
    )
    print("model =", model)

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.training.early_stopping
            )
        ],
        logger=True,
        enable_checkpointing=False,
        enable_model_summary=True,
    )

    # Lists for metrics
    train_losses, val_losses, val_mae, val_rmse = [], [], [], []

    with mlflow.start_run():
        # Log hyperparameters and git commit
        mlflow.log_params(
            {
                "back": cfg.preprocessing.back,
                "percent_of_training": cfg.preprocessing.percent_of_training,
                "scaler_min": cfg.preprocessing.scaler.feature_range.min,
                "scaler_max": cfg.preprocessing.scaler.feature_range.max,
                "hidden_size": cfg.model.hidden_size,
                "dropout": cfg.model.dropout,
                "lr": cfg.model.lr,
                "batch_size": cfg.training.batch_size,
                "max_epochs": cfg.training.max_epochs,
                "early_stopping": cfg.training.early_stopping,
                "git_commit": commit_id,
            }
        )

        mlflow.pytorch.autolog()
        trainer.fit(model, train_loader, val_loader)

        # Сохраняем модель
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(model.state_dict(), cfg.paths.model_save_path)
        X_af = X_train_all_feat
        mlflow.pytorch.log_model(model, "model", input_example=X_af)

        # Экспорт в ONNX
        if not os.path.exists("models"):
            os.mkdir("models")
        onnx_path = os.path.join("models", "lstm_model.onnx")
        dummy_input = torch.randn(
            1,
            X_train_all_feat.shape[1],
            X_train_all_feat.shape[2],
            device=model.device,
        )
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        mlflow.log_artifact(onnx_path)

        # Графики
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.title("Loss Curves")
        plt.savefig(f"{mlflow_cfg.mlflow.plots_dir}/loss_curves.png")
        mlflow.log_artifact(f"{mlflow_cfg.mlflow.plots_dir}/loss_curves.png")

        plt.figure()
        plt.plot(val_mae, label="Val MAE")
        plt.legend()
        plt.title("Validation MAE")
        plt.savefig(f"{mlflow_cfg.mlflow.plots_dir}/val_mae.png")
        mlflow.log_artifact(f"{mlflow_cfg.mlflow.plots_dir}/val_mae.png")

        plt.figure()
        plt.plot(val_rmse, label="Val RMSE")
        plt.legend()
        plt.title("Validation RMSE")
        plt.savefig(f"{mlflow_cfg.mlflow.plots_dir}/val_rmse.png")
        mlflow.log_artifact(f"{mlflow_cfg.mlflow.plots_dir}/val_rmse.png")

        # Логируем конфиг
        with open("configs/config.yaml", "r") as f:
            mlflow.log_artifact(f.name)


if __name__ == "__main__":
    main()
