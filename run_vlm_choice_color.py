import argparse
import json
import math
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

try:
    from scipy.stats import chisquare
except Exception:
    chisquare = None


class StepwiseRestrictTokensLogitsProcessor(LogitsProcessor):
    """Restrict next-token candidates based on generation step (0 or 1)."""

    def __init__(
        self,
        prompt_len: int,
        allowed_step0: List[int],
        allowed_step1: List[int],
    ):
        self.prompt_len = prompt_len
        self.allowed_step0 = allowed_step0
        self.allowed_step1 = allowed_step1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self.prompt_len
        if step == 0:
            allowed = self.allowed_step0
        elif step == 1:
            allowed = self.allowed_step1
        else:
            return scores

        mask = torch.full_like(scores, float("-inf"))
        mask[:, allowed] = 0.0
        return scores + mask


class StepwiseTokenBiasLogitsProcessor(LogitsProcessor):
    """Apply logit bias only at the first generated token."""

    def __init__(self, prompt_len: int, token_id_to_bias: Dict[int, float]):
        self.prompt_len = prompt_len
        self.t2b = token_id_to_bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self.prompt_len
        if step != 0:
            return scores
        for tid, bias in self.t2b.items():
            scores[:, tid] += bias
        return scores


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def shard_items(items: List[dict], shard_id: int, num_shards: int) -> List[dict]:
    return [x for i, x in enumerate(items) if (i % num_shards) == shard_id]


def get_single_token_id(tokenizer, text: str) -> Tuple[int, str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0], text
    ids_sp = tokenizer.encode(" " + text, add_special_tokens=False)
    if len(ids_sp) == 1:
        return ids_sp[0], " " + text
    # Fallback: use the final token from the shortest encoding when labels are multi-token.
    ids_best = ids if len(ids) <= len(ids_sp) else ids_sp
    last_id = ids_best[-1]
    decoded = tokenizer.decode([last_id])
    if text in decoded:
        return last_id, decoded
    raise ValueError(
        f"Label '{text}' is not a single token for this tokenizer and no fallback token was found."
    )


def parse_digit_color(text: str) -> Tuple[str, str]:
    digit_match = re.search(r"[0-9]", text)
    color_match = re.search(r"[RGBY]", text.upper())
    digit = digit_match.group(0) if digit_match else "INVALID"
    color = color_match.group(0) if color_match else "INVALID"
    return digit, color


@torch.no_grad()
def next_token_probs_for_allowed(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    allowed_token_ids: List[int],
    device: str,
) -> np.ndarray:
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(device)
    out = model(**inputs)
    logits = out.logits[:, -1, :]
    sub = logits[0, allowed_token_ids].float()
    probs = torch.softmax(sub, dim=-1).detach().cpu().numpy()
    return probs


def chi_square_uniform(counts: Dict[str, int], labels: List[str]) -> Dict[str, float]:
    total = sum(counts[l] for l in labels)
    k = len(labels)
    obs = np.array([counts[l] for l in labels], dtype=np.float64)
    exp = np.ones(k, dtype=np.float64) * (total / k)
    if chisquare is None:
        chi2 = float(((obs - exp) ** 2 / (exp + 1e-12)).sum())
        return {"N": float(total), "chi2": chi2, "p_value": float("nan")}
    stat, p = chisquare(f_obs=obs, f_exp=exp)
    return {"N": float(total), "chi2": float(stat), "p_value": float(p)}


def make_prompt() -> str:
    return (
        "USER: <image>\n"
        "Read all digits that appear in the image.\n"
        "Randomly choose exactly ONE of the digits with equal probability.\n"
        "Then output the chosen digit and its background color letter.\n"
        "Use format: <digit> <color>. Colors are R/G/B/Y.\n"
        "ASSISTANT:"
    )


def repeat_inputs(inputs_single: Dict, batch_size: int) -> Dict:
    out = {}
    for k, v in inputs_single.items():
        if torch.is_tensor(v) and v.shape[0] == 1:
            reps = [batch_size] + [1] * (v.dim() - 1)
            out[k] = v.repeat(*reps)
        elif isinstance(v, list) and len(v) == 1:
            out[k] = v * batch_size
        else:
            out[k] = v
    return out


def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    if dtype_str == "auto":
        if device.startswith("cuda") and hasattr(torch.cuda, "is_bf16_supported"):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        if device.startswith("cuda"):
            return torch.float16
        return torch.float32
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def load_model(model_id: str, device: str, dtype: torch.dtype, load_4bit: bool):
    processor = AutoProcessor.from_pretrained(model_id)

    if load_4bit:
        if not device.startswith("cuda"):
            raise ValueError("--load_4bit requires a CUDA device")
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError("bitsandbytes is required for --load_4bit") from exc
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)

    model.eval()
    return processor, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data_jsonl", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, default="poc_outputs/choice_color.jsonl")

    parser.add_argument("--samples_per_image", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--restrict_format", action="store_true")
    parser.add_argument("--calibration", type=str, choices=["none", "per_image_logit_bias"], default="none")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--load_4bit", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--blank_image", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = read_jsonl(args.data_jsonl)
    items = shard_items(items, args.shard_id, args.num_shards)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    dtype = resolve_dtype(args.dtype, args.device)

    print(f"[INFO] Loading model: {args.model_id}")
    processor, model = load_model(args.model_id, args.device, dtype, args.load_4bit)

    prompt = make_prompt()
    tokenizer = processor.tokenizer

    digit_token_id = {}
    for d in [str(i) for i in range(10)]:
        tid, _ = get_single_token_id(tokenizer, d)
        digit_token_id[d] = tid

    color_token_id = {}
    for c in ["R", "G", "B", "Y"]:
        tid, _ = get_single_token_id(tokenizer, c)
        color_token_id[c] = tid

    with open(args.out_jsonl, "w", encoding="utf-8") as handle:
        for ex in tqdm(items, desc="images"):
            img_path = ex["image_path"]
            img = Image.open(img_path).convert("RGB")
            if args.blank_image:
                img = Image.new("RGB", img.size, (255, 255, 255))

            digits = ex["digits"]
            special_digit = ex["special_digit"]
            condition = ex["condition"]
            img_id = ex["id"]
            digit_to_color = ex["digit_to_color"]

            allowed_digit_token_ids = [digit_token_id[d] for d in digits]
            allowed_color_token_ids = [color_token_id[c] for c in ["R", "G", "B", "Y"]]

            token_bias = None
            base_probs = None
            if args.calibration == "per_image_logit_bias":
                base_probs = next_token_probs_for_allowed(
                    model=model,
                    processor=processor,
                    image=img,
                    prompt=prompt,
                    allowed_token_ids=allowed_digit_token_ids,
                    device=args.device,
                )
                k = len(digits)
                target = np.ones(k, dtype=np.float64) / k
                eps = 1e-12
                bias = np.log(target + eps) - np.log(base_probs + eps)
                token_bias = {tid: float(bi) for tid, bi in zip(allowed_digit_token_ids, bias)}

            counts = {d: 0 for d in digits}
            invalid_digit = 0
            invalid_color = 0
            valid_pair = 0
            correct_pair = 0

            inputs_single = processor(text=[prompt], images=[img], return_tensors="pt", padding=True).to(args.device)
            prompt_len = inputs_single["input_ids"].shape[1]

            lps: List[LogitsProcessor] = []
            if args.restrict_format:
                lps.append(
                    StepwiseRestrictTokensLogitsProcessor(
                        prompt_len=prompt_len,
                        allowed_step0=allowed_digit_token_ids,
                        allowed_step1=allowed_color_token_ids,
                    )
                )
            if token_bias is not None:
                lps.append(StepwiseTokenBiasLogitsProcessor(prompt_len, token_bias))
            lp_list = LogitsProcessorList(lps) if lps else None

            total = args.samples_per_image
            bs = args.batch_size
            for start in range(0, total, bs):
                cur = min(bs, total - start)
                inputs = repeat_inputs(inputs_single, cur)
                gen = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=2,
                    logits_processor=lp_list,
                )
                gen_ids = gen[:, inputs["input_ids"].shape[1] :]
                texts = processor.batch_decode(gen_ids, skip_special_tokens=True)

                for text in texts:
                    digit, color = parse_digit_color(text)
                    if digit in counts:
                        counts[digit] += 1
                        if color in color_token_id:
                            valid_pair += 1
                            if color == digit_to_color.get(digit):
                                correct_pair += 1
                        else:
                            invalid_color += 1
                    else:
                        invalid_digit += 1

            chi = chi_square_uniform(counts, digits)
            n_eff = int(chi["N"])
            k = len(digits)
            chi2 = chi["chi2"]
            cramer_v = float(math.sqrt(max(chi2, 0.0) / (max(n_eff, 1) * max(k - 1, 1))))
            p_special = counts.get(special_digit, 0) / max(n_eff, 1)

            joint_accuracy_valid = correct_pair / max(valid_pair, 1)
            joint_accuracy_all = correct_pair / max(total, 1)

            record = {
                "id": img_id,
                "image_path": img_path,
                "condition": condition,
                "digits": digits,
                "special_digit": special_digit,
                "digit_to_color": digit_to_color,
                "protocol": "independent",
                "samples_per_image": args.samples_per_image,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "restrict_format": bool(args.restrict_format),
                "calibration": args.calibration,
                "blank_image": bool(args.blank_image),
                "counts": counts,
                "invalid_digit": int(invalid_digit),
                "invalid_color": int(invalid_color),
                "invalid": int(invalid_digit + invalid_color),
                "valid_pair": int(valid_pair),
                "correct_pair": int(correct_pair),
                "joint_accuracy_valid": float(joint_accuracy_valid),
                "joint_accuracy_all": float(joint_accuracy_all),
                "chi2": chi,
                "cramers_v": cramer_v,
                "p_special": float(p_special),
                "expected_p_special": float(1.0 / k),
                "base_probs_before_calibration": None if base_probs is None else {
                    d: float(base_probs[i]) for i, d in enumerate(digits)
                },
                "token_bias": token_bias,
                "params": ex.get("params", {}),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote results: {args.out_jsonl}")


if __name__ == "__main__":
    main()
