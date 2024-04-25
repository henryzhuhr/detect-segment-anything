import argparse
import os
from typing import List
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO


def get_args():
    DEFAULT_CPP = ".cache/weights/groundingdino_swint_ogc.pth"
    DEFAULT_TEXT_PROMPTS = ["person", "volleyball", "shoes", "ailurus fulgens", "panda"]
    parser = argparse.ArgumentParser(description="GroundingDINO Inference")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, cpu")
    parser.add_argument("--config_file", type=str, default="config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CPP)
    parser.add_argument("--img_path", type=str, default="examples/ailurus_fulgens.jpg")
    parser.add_argument("--text_prompts", metavar="N", type=str, nargs="+", default=DEFAULT_TEXT_PROMPTS)
    parser.add_argument("--box_theshold", type=float, default=0.35)
    parser.add_argument("--text_theshold", type=float, default=0.25)
    parser.add_argument("--export_types", metavar="N", type=str, nargs="+", default=["openvino"])
    parser.add_argument("--export_dir", type=str, default="tmp")
    parser.add_argument("--export_name", type=str, default="groundingDINO")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cpu")  # using CPU to export to avoid bug
    config_file: str = args.config_file
    checkpoint_path: str = args.checkpoint_path
    export_dir: str = args.export_dir
    export_name: str = args.export_name

    model: GroundingDINO = load_model(config_file, checkpoint_path, device, is_export=True)
    export(model, export_dir, export_name, device)


def export(model: GroundingDINO, export_dir="tmp", export_name="model", device="cpu"):
    """
    see:
        1. https://pytorch.org/tutorials//beginner/onnx/export_simple_model_to_onnx_tutorial.html
        2. https://pytorch.org/docs/stable/onnx_torchscript.html
    refer OpenVINO https://blog.openvino.ai/blog-posts/enable-openvino-tm-optimization-for-groundingdino
    """
    export_file = os.path.join(export_dir, f"{export_name}.onnx")

    tokenizer = model.tokenizer
    tokenized = tokenizer.__call__(["the running dog ."], return_tensors="pt")
    input_ids: torch.Tensor = tokenized["input_ids"]
    token_type_ids = tokenized["token_type_ids"]
    attention_mask = tokenized["attention_mask"]
    position_ids = torch.arange(input_ids.shape[1]).reshape(1, -1)
    text_token_mask = torch.randint(0, 2, (1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.bool)

    position_ids = torch.tensor([[0, 0, 1, 2, 3, 0]])
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[True, True, True, True, True, True]])
    # fmt: off
    text_token_mask = torch.tensor([[
        [True, False, False, False, False, False],
        [False, True, True, True, True, False],
        [False, True, True, True, True, False],
        [False, True, True, True, True, False],
        [False, True, True, True, True, False],
        [False, False, False, False, False, True],
    ]])
    # fmt: on

    img = torch.randn(1, 3, 800, 800, device=device)

    # export DINO
    input_names = [
        "img",
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "text_token_mask",
    ]
    dummpy_inputs = (
        img,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids,
        text_token_mask,
    )
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
    output_names = ["logits", "boxes"]
    for par in model.parameters():
        par.requires_grad = False
    # If we don't trace manually ov.convert_model will try to trace it automatically with default check_trace=True, which fails.
    # Therefore we trace manually with check_trace=False, despite there are warnings after tracing and conversion to OpenVINO IR
    # output boxes are correct.
    # traced_model = torch.jit.trace(
    #     model,
    #     example_inputs=dummpy_inputs,
    #     strict=False,
    #     check_trace=False,
    # )
    torch.onnx.export(
        model,
        dummpy_inputs,
        export_file,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16,  # issue: https://github.com/pytorch/pytorch/issues/104190#issuecomment-1607676629
    )

    try:
        import openvino as ov

        is_export_ov = True
    except ImportError:
        is_export_ov = False

    if is_export_ov:
        # export OpenVINO
        # core = ov.Core()
        # ov_input_names = {  # https://docs.openvino.ai/2024/api/ie_python_api/_autosummary/openvino.Type.html
        #     "samples": ov.Type.f32,
        #     "input_ids": ov.Type.i64,
        #     "attention_mask": ov.Type.boolean,
        #     "position_ids": ov.Type.i64,
        #     "token_type_ids": ov.Type.i64,
        #     "text_self_attention_masks": ov.Type.boolean,
        # }
        ov_model = ov.convert_model(model, example_input=dummpy_inputs)
        ov.save_model(ov_model, export_file.replace(".onnx", ".xml"))


if __name__ == "__main__":
    main()
