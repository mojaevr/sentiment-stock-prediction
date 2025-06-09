# Stock-Market-Prediction-Using-LSTM-and-Online-News-Sentiment-Analysis

## Постановка задачи

Задача заключается в создании системы, которая предсказывает цены на акции на основе исторических данных и анализа новостей. Это необходимо для помощи инвесторам и трейдерам в принятии решений, а также для автоматизации торговых стратегий. Система будет использовать нейронные сети для анализа временных рядов и обработки текстовых
данных из новостей, чтобы учитывать влияние событий на рынок.

## Формат входных и выходных данных

### Входные данные:

1. Исторические данные по ценам акций (например, открытие, закрытие, максимум,
минимум, объем торгов) в формате временных рядов.

Размерность: (N, T, F), где N — количество акций, T — временные шаги, F —
количество признаков (например, 5 для OHLCV).

2. Текстовые данные новостей, связанных с компаниями или рынком.
Размерность: (N, T, S), где S — длина текста (после предобработки).

### Выходные данные:
1. Прогнозируемая цена акции на следующий временной шаг (например, цена закрытия)
Размерность: (N, 1).

## Метрики

1. Mean Absolute Error (MAE) — измеряет среднюю абсолютную ошибку в предсказаниях цен в долларах. Целевое значение: MAE < 1-2% от средней цены актива. Это выбрано, так как MAE интерпретируема в реальных единицах и важна
для финансовых приложений.

2. Root Mean Squared Error (RMSE) — штрафует большие отклонения сильнее, что полезно для оценки стабильности модели. Целевое значение: RMSE < 2-3% от средней цены.

3. Accuracy (точность направления) — процент случаев, когда модель правильно предсказала направление изменения цены. Ожидаемое значение: 60-70%.


## Описание данных
Исторические данные по акциям:

Источник: Yahoo Finance.

Особенности: Данные могут содержать пропуски, шумы или аномалии.

Новостные данные:

Источник: Reuters,

Особенности: Тексты могут быть неструктурированными, содержать шум (например, рекламу).

## Описание модели
Архитектура:

Для временных рядов: LSTM.

Для текстовых данных: BERT для извлечения эмбеддингов новостей.

Объединение: Concatenate эмбеддингов новостей с выходом LSTM, затем
полносвязные слои для предсказания.

_________

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/mojaevr/sentiment-stock-prediction.git
cd Stock-Market-Prediction-Using-LSTM-and-Online-News-Sentiment-Analysis
```

2. **Install Poetry** (if not installed):

```bash
pip install poetry
```

3. **Install dependencies**

```bash
poetry install
```

4. **Activate the virtual environment**
```bash
env activate
```

5. **Install pre-commit hooks**

```bash
poetry run pre-commit install
```

6. **Check code style and auto-fix**

```bash
poetry run pre-commit run -a
```

7. **Prepare data**

```bash
dvc init
dvc remote add -d mygdrive gdrive://15ZBdfSVrRd_ih5VrdJvFd07oA-3HcinJ
dvc remote modify mygdrive gdrive_use_service_account true
dvc remote modify mygdrive --local gdrive_service_account_json_file_path cred.json
dvc pull
```

8. **Run training**

```bash
python train_lstm_lightning.py
```

9. **Run evaluation**

```bash
python evaluate_lstm_lightning.py
```

10. **Plot predictions**

```bash
python plot_lstm_lightning.py
```

---

## Train

### 1. Data Preparation

- Download the folder `_data sets_` using dvc

### 2. Preprocessing (optional)

- You can adjust preprocessing parameters in `configs/preprocessing.yaml`.
- The main training script will automatically apply preprocessing as defined in the configs.

### 3. Model Training

To train the LSTM model, run:

```bash
python train_lstm_lightning.py
```

- Training parameters (batch size, epochs, early stopping, etc.) can be changed in `configs/training.yaml`.
- Model architecture parameters can be changed in `configs/model.yaml` (if present).
- All experiment logs and artifacts will be saved via MLflow (see `configs/mlflow.yaml`).

### 4. Model Evaluation

To evaluate the trained model:

```bash
python evaluate_lstm_lightning.py
```

### 5. Plotting Predictions

To visualize predictions:

```bash
python plot_lstm_lightning.py
```

---

## Production preparation

### 1. Model Export

After training, the model is automatically exported to ONNX format for production use. The ONNX file is saved to:

```
models/lstm_model.onnx
```

You can use this file for inference in any ONNX-compatible runtime (e.g., ONNX Runtime, TensorRT, OpenVINO, etc.).

#### To export the model manually (if needed):

```bash
python train_lstm_lightning.py  # The script will export the ONNX file after training
```

### 2. Model Artifacts

The following artifacts are required for production deployment:

- `models/lstm_model.pth` — PyTorch weights (for further fine-tuning or research)
- `models/lstm_model.onnx` — ONNX model for production inference
- `configs/config.yaml` — Main config with all paths and defaults
- `configs/preprocessing.yaml` — Preprocessing parameters
- `configs/model.yaml` — Model architecture parameters (if present)
- `configs/training.yaml` — Training parameters
- `configs/mlflow.yaml` — MLflow experiment tracking config
- `data sets/data_to_paper_microsoft_case.pkl` — Final dataset used for inference

### 3. Optional: Further Optimization

- You can further optimize the ONNX model using tools like ONNX Runtime or convert it to TensorRT for GPU inference:

```bash
# Example: optimize ONNX model with onnxruntime-tools
pip install onnxruntime-tools
python -m onnxruntime_tools.optimizer_cli --input models/lstm_model.onnx --output models/lstm_model_optimized.onnx
```

- For TensorRT conversion, refer to NVIDIA documentation for your platform.

---

## Infer

After training, you can use the exported ONNX model for fast inference on new data. This section describes how to run inference and the required input format.

### 1. Input Data Format

- The model expects a time series of Microsoft stock prices and an additional feature (e.g., sentiment or open price) as a NumPy array of shape `(N, 2)`, where `N` is the sequence length.
- Data must be preprocessed in the same way as during training (scaling, look-back window, etc.).
- Example input (CSV):

```csv
MSFT,open
305.22,304.50
306.10,305.00
307.00,306.20
...
```

- Example code to prepare input for ONNX inference:

```python
import numpy as np
import pandas as pd
import onnxruntime as ort

# Load and preprocess your data (example)
df = pd.read_csv('your_new_data.csv')
# Apply the same scaling and look-back as in training
# ...
X = np.stack([feature1, feature2], axis=1)  # shape: (seq_len, 2)
X = X[np.newaxis, ...]  # add batch dimension: (1, seq_len, 2)

# Run inference
session = ort.InferenceSession('models/lstm_model.onnx')
outputs = session.run(None, {"input": X.astype(np.float32)})
prediction = outputs[0]
print(prediction)
```

### 2. Minimal Dependencies for Inference

- For ONNX inference, only `onnxruntime`, `numpy`, and `pandas` are required:

```bash
pip install onnxruntime numpy pandas
```

### 3. Example Artifact

- See `data sets/example_infer_input.csv` for a sample input file.
- The output will be a NumPy array with predicted values for the given sequence.

---

* Before running the final notebook, please download the folder _data sets_ and create the final data set with the notebook _04 data wrangling_.

If you find any issue with the notebooks, please contact camiloegomeznarvaez@gmail.com
