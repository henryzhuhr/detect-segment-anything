import argparse
import os
import time
from typing import List
import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate


def get_args():

    parser = argparse.ArgumentParser(description="GroundingDINO Inference")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, cpu")
    parser.add_argument(
        "--config_file", type=str, default="config/GroundingDINO_SwinT_OGC.py"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=".cache/weights/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--img_path", type=str, default="examples/cjy.jpg")
    parser.add_argument(
        "--text_prompts", type=List[str], default=["person", "volleyball", "shoes"]
    )
    parser.add_argument("--box_theshold", type=float, default=0.35)
    parser.add_argument("--text_theshold", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = get_args()
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.cuda.is_mps_supported() else "cpu"
        )
    print(f"Using device: {device}")
    exit()
    config_file: str = args.config_file
    checkpoint_path: str = args.checkpoint_path
    img_path: str = args.img_path
    text_prompts: str = args.text_prompts
    box_theshold: float = args.box_theshold
    text_theshold: float = args.text_theshold

    model: torch.nn.Module = load_model(config_file, checkpoint_path, device)
    image_source, image = load_image(img_path)

    for i in range(10):
        st = time.time()
        caption = ". ".join(text_prompts)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=caption,
            box_threshold=box_theshold,
            text_threshold=text_theshold,
            device=device,
        )
        ut = (time.time() - st) * 1000
        print(f"Time taken: {ut}ms")
    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    )
    os.makedirs(save_dir := "tmp", exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "annotated_image.jpg"), annotated_frame)


if __name__ == "__main__":
    main()
