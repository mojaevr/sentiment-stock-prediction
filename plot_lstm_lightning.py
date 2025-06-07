# plot_lstm_lightning.py
"""
Plot predictions of the trained LSTM model from PyTorch Lightning.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler

from train_lstm_lightning import LSTMRegressor, create_dataset

sns.set_context("paper", font_scale=1.3)
sns.set_style("white")

# Data loading and preprocessing (same as in training)
result = pd.read_pickle("data sets/data_to_paper_microsoft_case.pkl")
y = result.MSFT.values.astype("float32").reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
y = scaler.fit_transform(y)
X = result.open.values.astype("float32").reshape(-1, 1)
percent_of_training = 0.7
train_size = int(len(y) * percent_of_training)
train_y, test_y = y[:train_size], y[train_size:]
train_x, test_x = X[:train_size], X[train_size:]
back = 7
X_train_feat_1, y_train = create_dataset(train_y, back)
X_test_feat_1, y_test = create_dataset(test_y, back)
X_train_feat_2, _ = create_dataset(train_x, back)
X_test_feat_2, _ = create_dataset(test_x, back)
X_train_feat_1 = np.reshape(X_train_feat_1, (X_train_feat_1.shape[0], back))
X_test_feat_1 = np.reshape(X_test_feat_1, (X_test_feat_1.shape[0], back))
X_train_feat_2 = np.reshape(X_train_feat_2, (X_train_feat_2.shape[0], back))
X_test_feat_2 = np.reshape(X_test_feat_2, (X_test_feat_2.shape[0], back))
# Формируем shape (samples, seq_len, feat=2)
X_train_all_feat = np.stack([X_train_feat_1, X_train_feat_2], axis=-1)
X_test_all_feat = np.stack([X_test_feat_1, X_test_feat_2], axis=-1)

# Load model
input_size = X_train_all_feat.shape[2]
model = LSTMRegressor(input_size=input_size)
model.load_state_dict(
    torch.load("lstm_lightning_model.pth", map_location=torch.device("cpu"))
)
model.eval()

X_train = torch.tensor(X_train_all_feat, dtype=torch.float32)
X_test = torch.tensor(X_test_all_feat, dtype=torch.float32)

with torch.no_grad():
    train_pred = model(X_train).cpu().numpy()
    test_pred = model(X_test).cpu().numpy()

# Prepare DataFrames for plotting
start = back + 1
time_y_train = pd.DataFrame(
    data=y_train, index=result[0:train_size].index[start:], columns=[""]
)
time_y_test = pd.DataFrame(
    data=y_test, index=result[train_size:].index[start:], columns=[""]
)
time_y_train_prediction = pd.DataFrame(
    data=train_pred, index=time_y_train.index, columns=[""]
)
time_y_test_prediction = pd.DataFrame(
    data=test_pred, index=time_y_test.index, columns=[""]
)

plt.figure(figsize=(15, 10))
plt.plot(time_y_train, label="training", color="green", marker=".")
plt.plot(time_y_test, label="test", marker=".")
plt.plot(time_y_train_prediction, color="red", label="prediction")
plt.plot(time_y_test_prediction, color="red")
plt.title(
    "LSTM fit of Microsoft Stock Market Prices Including Sentiment Signal",
    size=20,
)
plt.tight_layout()
sns.despine(top=True)
plt.ylabel("", size=15)
plt.xlabel("", size=15)
plt.legend(fontsize=15)
plt.grid()
plt.show()
