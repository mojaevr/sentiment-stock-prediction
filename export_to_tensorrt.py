import argparse

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_file_path, engine_file_path, max_batch_size=1):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = max_batch_size
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        engine = builder.build_engine(network, config)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print(f"TensorRT engine saved to {engine_file_path}")
        return engine


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to TensorRT engine."
    )
    parser.add_argument(
        "--onnx", type=str, required=True, help="Path to ONNX model file"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="models/lstm_model.trt",
        help="Path to save TensorRT engine",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    build_engine(args.onnx, args.engine, args.max_batch_size)


if __name__ == "__main__":
    main()
