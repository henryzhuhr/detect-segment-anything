import argparse
import os
import time
from typing import List
import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO


def get_args():

    parser = argparse.ArgumentParser(description="GroundingDINO Inference")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, cpu")
    parser.add_argument("--config_file", type=str, default="config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=".cache/weights/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--img_path", type=str, default="examples/ailurus_fulgens.jpg")
    parser.add_argument(
        "--text_prompts",
        type=List[str],
        default=["person", "volleyball", "shoes", "ailurus fulgens", "panda"],
    )
    parser.add_argument("--box_theshold", type=float, default=0.35)
    parser.add_argument("--text_theshold", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cpu")
    config_file: str = args.config_file
    checkpoint_path: str = args.checkpoint_path

    model: GroundingDINO = load_model(config_file, checkpoint_path, device, is_export=True)

    dummy_input = torch.randn(1, 3, 800, 800, device=device)
    export_dir = "tmp"
    export_name = "groundingDINO"
    export_onnx(model, dummy_input, export_dir, export_name)


def export_onnx(
    model: GroundingDINO,
    dummy_input: torch.Tensor,
    export_dir="tmp",
    export_name="groundingDINO",
):
    """
    see:
        1. https://pytorch.org/tutorials//beginner/onnx/export_simple_model_to_onnx_tutorial.html
        2. https://pytorch.org/docs/stable/onnx_torchscript.html
    refer OpenVINO https://blog.openvino.ai/blog-posts/enable-openvino-tm-optimization-for-groundingdino
    """
    export_file = os.path.join(export_dir, f"{export_name}.onnx")
    if os.path.exists(export_file):
        os.remove(export_file)

    # prepare dummy inputs
    caption = "the running dog ."  # ". ".join(input_text)

    input_ids = model.tokenizer([caption], return_tensors="pt")[
        "input_ids"
    ]  # https://huggingface.co/docs/transformers/main_classes/onnx
    position_ids = torch.tensor([[0, 0, 1, 2, 3, 0]])
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[True, True, True, True, True, True]])
    text_token_mask = torch.tensor(
        [
            [
                [True, False, False, False, False, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, False, False, False, False, True],
            ]
        ]
    )
    images = torch.randn(1, 3, 800, 800)
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
        "token_type_ids": {0: "batch_size", 1: "seq_len"},
        "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
        "img": {0: "batch_size", 2: "height", 3: "width"},
        "logits": {0: "batch_size"},
        "boxes": {0: "batch_size"},
    }
    inputs = {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_type_ids": token_type_ids,
        "text_token_mask": text_token_mask,
    }
    input_names = [
        "images",
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "text_token_mask",
    ]
    output_names = ["logits", "boxes"]
    torch.onnx.export(
        model,
        (
            images,
            input_ids,
            attention_mask,
            position_ids,
            token_type_ids,
            text_token_mask,
        ),
        export_file,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16,
    )


if __name__ == "__main__":
    main()
