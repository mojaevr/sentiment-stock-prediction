# evaluate_lstm_lightning.py
"""
Evaluate trained LSTM model from PyTorch Lightning and print metrics.
"""
# DVC: Ensure data is pulled from remote (Google Drive)
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from train_lstm_lightning import LSTMRegressor, create_dataset

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
look_back = 7
X_train_f_1, y_train = create_dataset(train_y, look_back)
X_test_f_1, y_test = create_dataset(test_y, look_back)
X_train_f_2, _ = create_dataset(train_x, look_back)
X_test_f_2, _ = create_dataset(test_x, look_back)
X_train_f_1 = np.reshape(X_train_f_1, (X_train_f_1.shape[0], look_back))
X_test_f_1 = np.reshape(X_test_f_1, (X_test_f_1.shape[0], look_back))
X_train_f_2 = np.reshape(X_train_f_2, (X_train_f_2.shape[0], look_back))
X_test_f_2 = np.reshape(X_test_f_2, (X_test_f_2.shape[0], look_back))
# Формируем shape (samples, seq_len, f=2)
X_train_all_f = np.stack([X_train_f_1, X_train_f_2], axis=-1)
X_test_all_f = np.stack([X_test_f_1, X_test_f_2], axis=-1)

# Load model
input_size = X_train_all_f.shape[2]
model = LSTMRegressor(input_size=input_size)
model.load_state_dict(
    torch.load("lstm_lightning_model.pth", map_location=torch.device("cpu"))
)
model.eval()

X_train = torch.tensor(X_train_all_f, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test_all_f, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

with torch.no_grad():
    train_pred = model(X_train).cpu().numpy()
    test_pred = model(X_test).cpu().numpy()

print("Train Mean Absolute Error:", mean_absolute_error(y_train, train_pred))
print(
    "Train Root Mean Squared Error:",
    np.sqrt(mean_squared_error(y_train, train_pred)),
)
print("Test Mean Absolute Error:", mean_absolute_error(y_test, test_pred))
print(
    "Test Root Mean Squared Error:",
    np.sqrt(mean_squared_error(y_test, test_pred)),
)
