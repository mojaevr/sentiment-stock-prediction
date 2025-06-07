# Triton Inference Server для LSTM ONNX модели

## 1. Установка Triton Inference Server (на Linux)

Triton поддерживается только на Linux и требует Docker. Установите Docker, если он ещё не установлен.

```bash
# Установите Docker (если не установлен)
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

## 2. Подготовка модели

ONNX-модель должна быть размещена в структуре каталогов Triton:

```
triton/model_repository/lstm/
  ├── 1/
  │   └── model.onnx
  └── config.pbtxt
```

- Поместите ваш onnx-файл (например, `models/lstm_model.onnx`) в `triton/model_repository/lstm/1/model.onnx`.
- Создайте файл `config.pbtxt` (пример ниже).

### Пример config.pbtxt для LSTM ONNX модели

```
name: "lstm"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 7, 2 ]  # [seq_len, features] - скорректируйте под вашу модель
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]
```

## 3. Запуск Triton Inference Server

```bash
docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.03-py3 \
  tritonserver --model-repository=/models
```

- Для CPU можно использовать тег `tritonserver:24.03-py3` без `--gpus all`.

## 4. Инференс через HTTP API

Пример запроса через Python:

```python
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")
input_data = np.random.rand(1, 7, 2).astype(np.float32)  # (batch, seq_len, features)
inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)
outputs = [httpclient.InferRequestedOutput("output")]
response = client.infer("lstm", inputs, outputs=outputs)
print(response.as_numpy("output"))
```

## 5. DVC

ONNX-файл и пример данных для инференса рекомендуется добавить под DVC:

```bash
dvc add models/lstm_model.onnx
```

---

# Infer: MLflow Model Serving

## 1. Запуск сервера

1. Найдите путь к модели в MLflow UI (http://127.0.0.1:8080):
   - Откройте эксперимент, выберите нужный запуск (run), скопируйте путь к артефакту model (например, mlruns/0/1234567890abcdef/artifacts/model)

2. Запустите сервер:

```bash
mlflow models serve -m "mlruns/<experiment_id>/<run_id>/artifacts/model" -p 5001 --host 127.0.0.1
```

## 2. Пример инференса через REST API

```python
import requests
import numpy as np
import json

input_data = np.random.rand(1, 7, 2).tolist()  # (batch, seq_len, features)
payload = {"inputs": [
    {"name": "input", "shape": [1, 7, 2], "datatype": "FP32", "data": input_data}
]}
headers = {"Content-Type": "application/json"}
response = requests.post("http://127.0.0.1:5001/invocations", data=json.dumps(payload), headers=headers)
print(response.json())
```

- Для других форматов входа см. [MLflow docs](https://www.mlflow.org/docs/latest/models.html#deploy-mlflow-models)

---
