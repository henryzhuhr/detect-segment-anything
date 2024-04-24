import argparse
import os
import time
from typing import Dict, List, Tuple
import cv2
import torch
from groundingdino.util.inference import load_model, load_image, annotate

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap

import bisect
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
    device = torch.device(
        args.device
        if args.device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
            # TODO: Add support for MPS. torch.backends.mps.is_available()
        )
    )
    config_file: str = args.config_file
    checkpoint_path: str = args.checkpoint_path
    img_path: str = args.img_path
    text_prompts: str = args.text_prompts
    box_theshold: float = args.box_theshold
    text_theshold: float = args.text_theshold

    model: GroundingDINO = load_model(config_file, checkpoint_path, device)

    image_source, image = load_image(img_path)

    for i in range(1):
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
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    os.makedirs(save_dir := "tmp", exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "annotated_image.jpg"), annotated_frame)


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def predict(
    model: GroundingDINO,
    image: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str = "cuda",
    remove_combined: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    images = image.to(device).unsqueeze(0)

    captions = [caption]

    # encoder captions
    tokenized: Dict[str, torch.Tensor] = model.tokenizer.__call__(
        captions, padding="longest", return_tensors="pt"
    ).to(device)
    print(tokenized)
    exit()
    specical_tokens = model.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
    ) = generate_masks_with_special_tokens_and_transfer_map(tokenized, specical_tokens, model.tokenizer)
    if text_self_attention_masks.shape[1] > model.max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, : model.max_text_len, : model.max_text_len]
        position_ids = position_ids[:, : model.max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : model.max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : model.max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : model.max_text_len]
    with torch.no_grad():
        outputs = model(
            image[None],
            tokenized["input_ids"],
            tokenized["attention_mask"],
            position_ids,
            tokenized["token_type_ids"],
            text_self_attention_masks,
        )
    # exit()
    # return None

    # with torch.no_grad():
    #     outputs = model(images, captions=[caption])
    # # exit()

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    if remove_combined:
        sep_idx = [
            i for i in range(len(tokenized["input_ids"])) if tokenized["input_ids"][i] in [101, 102, 1012]
        ]

        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(
                get_phrases_from_posmap(
                    logit > text_threshold, tokenized, tokenizer, left_idx, right_idx
                ).replace(".", "")
            )
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace(".", "")
            for logit in logits
        ]

    return boxes, logits.max(dim=1)[0], phrases


def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list, tokenizer):
    """Generate attention mask between each pair of special tokens
    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1)
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
            c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col

    cate_to_token_mask_list = [
        torch.stack(cate_to_token_mask_listi, dim=0) for cate_to_token_mask_listi in cate_to_token_mask_list
    ]

    # # padding mask
    # padding_mask = tokenized['attention_mask']
    # attention_mask = attention_mask & padding_mask.unsqueeze(1).bool() & padding_mask.unsqueeze(2).bool()

    return attention_mask, position_ids.to(torch.long), cate_to_token_mask_list


if __name__ == "__main__":
    main()
