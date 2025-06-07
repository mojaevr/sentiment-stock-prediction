#!/bin/bash
# Usage: bash export_to_tensorrt.sh models/lstm_model.onnx models/lstm_model.trt
set -e
ONNX_PATH=${1:-models/lstm_model.onnx}
TRT_PATH=${2:-models/lstm_model.trt}
python export_to_tensorrt.py --onnx "$ONNX_PATH" --engine "$TRT_PATH"
