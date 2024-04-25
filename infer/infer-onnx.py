import argparse
import os
import time
from typing import List
import cv2
import numpy as np
import onnxruntime as rt
import torch
from transformers import AutoTokenizer, BertTokenizerFast
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.inference import annotate


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_path", default="tmp/groundingDINO.onnx")
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument(
        "--img_path",
        default="examples/cjy.jpg",
    )
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_arguments()
    model_path: str = args.model_path
    max_text_len: int = args.max_text_len
    box_threshold: float = args.box_threshold
    text_threshold: float = args.text_threshold

    text_prompts = ["person", "volleyball", "shoes", "ailurus fulgens", "panda"]
    # 为 text_prompts 每个类别随机生成一个颜色
    color_map = []
    for i in range(len(text_prompts)):
        color_map.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(
        "bert-base-uncased", cache_dir=".cache", local_files_only=True
    )

    so = rt.SessionOptions()
    # session = rt.InferenceSession(
    #     model_path,
    #     so,
    #     providers=["OpenVINOExecutionProvider"],
    #     provider_options=[{"device_type": "CPU_FP32"}],
    # )
    device = args.device
    providers = (
        [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        if device != "cpu"
        else ["CPUExecutionProvider"]
    )
    session = rt.InferenceSession(model_path, so, providers=providers)

    input_names = session.get_inputs()
    outputs = session.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    # for name in input_names:
    #     print(name.name, name.shape, name.type)

    # print(output_names)

    img_src = cv2.imread(args.img_path)
    img = cv2.resize(img_src, (800, 800))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img = np.ascontiguousarray(img)
    print(img.shape)

    caption = ". ".join(text_prompts)
    captions = [preprocess_caption(caption=caption)]

    tokenized = tokenizer.__call__(captions, padding="longest", return_tensors="np")
    # for key, value in tokenized.items():
    #     print(key, value.shape)
    specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
    ) = generate_masks_with_special_tokens_and_transfer_map_np(tokenized, specical_tokens)

    st = time.time()
    outputs = session.run(
        output_names,
        {
            "images": img,
            "input_ids": tokenized["input_ids"],
            "attention_mask": (tokenized["attention_mask"]).astype(np.bool_),
            "position_ids": position_ids[:, :max_text_len],
            "token_type_ids": tokenized["token_type_ids"],
            "text_token_mask": text_self_attention_masks[:, :max_text_len, :max_text_len],
        },
    )
    pred_logits: np.ndarray = outputs[0][0]
    pred_logits = sigmoid(pred_logits)
    pred_boxes: np.ndarray = outputs[1][0]

    mask = pred_logits.max(axis=1) > box_threshold
    logits = pred_logits[mask]
    boxes = pred_boxes[mask]
    tokenized = tokenizer(caption)

    phrases = []
    left_idx: int = 0
    right_idx: int = 255
    for logit in logits:
        posmap = logit > text_threshold
        posmap[0 : left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = np.where(posmap)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        phrase = tokenizer.decode(token_ids).replace(".", "")
        phrases.append(phrase)

    logits = logits.max(axis=1)
    # boxes=torch.from_numpy(boxes)
    # logits=torch.from_numpy(logits)
    # print(phrases)

    h, w, _ = img_src.shape
    boxes_l = (boxes * np.array([w, h, w, h])).astype(int).tolist()
    scene = img_src.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale: float = 0.5
    text_thickness: int = 1
    text_padding: int = 5
    for i, phrase in enumerate(phrases):
        print(phrase, boxes_l[i], logits[i])
        cx, cy, w, h = boxes_l[i]
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = cx + w // 2
        y2 = cy + h // 2
        color = color_map[text_prompts.index(phrase)]

        cv2.rectangle(scene, (x1, y1), (x2, y2), color, text_thickness * 2)
        text = f"{phrase} {logits[i]:.2f}"

        text_width, text_height = cv2.getTextSize(
            text=text,
            fontFace=font,
            fontScale=text_scale,
            thickness=text_thickness,
        )[0]
        text_x = x1 + text_padding
        text_y = y1 - text_padding

        # text_background
        tb_x1 = x1
        tb_y1 = y1 - 2 * text_padding - text_height

        tb_x2 = x1 + 2 * text_padding + text_width
        tb_y2 = y1

        cv2.rectangle(
            img=scene,
            pt1=(tb_x1, tb_y1),
            pt2=(tb_x2, tb_y2),
            color=color,
            thickness=cv2.FILLED,
        )

        cv2.putText(
            img=scene,
            text=text,
            org=(text_x, text_y),
            fontFace=font,
            fontScale=text_scale,
            color=(255, 255, 255),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    # annotated_frame = annotate(image_source=img_src, boxes=boxes, logits=logits, phrases=phrases)

    # print("Time taken: ", time.time()-st)
    os.makedirs(save_dir := "tmp", exist_ok=True)
    # annotated_frame = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, "annotated_image.onnx.jpg"), scene)


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_masks_with_special_tokens_and_transfer_map_np(tokenized, special_tokens_list):
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
    special_tokens_mask = np.zeros((bs, num_token)).astype(np.bool_)

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = np.stack(np.nonzero(special_tokens_mask), axis=1)

    # generate attention mask and positional ids
    attention_mask = np.eye(num_token).astype(np.bool_).reshape(1, num_token, num_token).repeat(bs, 1)
    position_ids = np.zeros((bs, num_token)).astype(np.float32)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = np.arange(0, col - previous_col)
            c2t_maski = np.zeros((num_token)).astype(np.bool_)
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col

    # cate_to_token_mask_list = [
    #     np.stack(cate_to_token_mask_listi, dim=0) for cate_to_token_mask_listi in cate_to_token_mask_list
    # ]
    cate_to_token_mask_list = []
    for cate_to_token_mask_listi in cate_to_token_mask_list:
        a_s = np.stack(cate_to_token_mask_listi, axis=0)
        cate_to_token_mask_list.append(a_s)

    # # padding mask
    # padding_mask = tokenized['attention_mask']
    # attention_mask = attention_mask & padding_mask.unsqueeze(1).bool() & padding_mask.unsqueeze(2).bool()

    return attention_mask, position_ids.astype(np.int64), cate_to_token_mask_list


if __name__ == "__main__":
    main()
