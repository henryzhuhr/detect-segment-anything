import argparse
from typing import List
import cv2
import numpy as np
import onnxruntime as rt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_path", default="tmp/groundingDINO.onnx")
    parser.add_argument(
        "--img_path",
        default="examples/cjy.jpg",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    model_path: str = args.model_path

    so = rt.SessionOptions()
    # session = rt.InferenceSession(
    #     model_path,
    #     so,
    #     providers=["OpenVINOExecutionProvider"],
    #     provider_options=[{"device_type": "CPU_FP32"}],
    # )

    session = rt.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])

    input_names = session.get_inputs()
    for name in input_names:
        print(name.name, name.shape, name.type)
    outputs = session.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    print(output_names)

    img = cv2.imread(args.img_path)
    img = cv2.resize(img, (800, 800))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img = np.ascontiguousarray(img)
    print(img.shape)

    session.run(
        output_names,
        {
            "images": img,
        },
    )

def sig(x):
    return 1/(1 + np.exp(-x))

if __name__ == "__main__":
    main()
