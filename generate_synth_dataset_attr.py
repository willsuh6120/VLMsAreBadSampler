import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageFilter


COLOR_LETTER_TO_RGB: Dict[str, Tuple[int, int, int]] = {
    "R": (235, 90, 90),
    "G": (90, 210, 120),
    "B": (90, 130, 235),
    "Y": (240, 210, 90),
}


def load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Load a font at the requested size, fall back to default if unavailable."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, font_size)
    return ImageFont.load_default()


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def blend_rgb(rgb: Tuple[int, int, int], gray_level: int, alpha: float) -> Tuple[int, int, int]:
    """Blend a color toward gray to reduce contrast/saturation."""
    alpha = clamp01(alpha)
    r, g, b = rgb
    rr = int(round((1.0 - alpha) * r + alpha * gray_level))
    gg = int(round((1.0 - alpha) * g + alpha * gray_level))
    bb = int(round((1.0 - alpha) * b + alpha * gray_level))
    return rr, gg, bb


def draw_digit_in_cell(
    draw: ImageDraw.ImageDraw,
    cell_box: Tuple[int, int, int, int],
    digit: str,
    font: ImageFont.ImageFont,
    bg_color: Tuple[int, int, int],
) -> None:
    """Draw a single digit centered inside a colored cell."""
    x0, y0, x1, y1 = cell_box
    draw.rectangle(cell_box, fill=bg_color, outline=(0, 0, 0), width=3)

    bbox = draw.textbbox((0, 0), digit, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    tx = cx - text_w // 2
    ty = cy - text_h // 2

    draw.text(
        (tx, ty),
        digit,
        font=font,
        fill=(0, 0, 0),
        stroke_width=2,
        stroke_fill=(255, 255, 255),
    )


def make_image_and_meta(
    img_size: int,
    condition: str,
    digits: List[str],
    seed: int,
) -> Tuple[Image.Image, Dict]:
    """Create a synthetic image and metadata for a given condition."""
    random.seed(seed)

    grid = 3
    cell = img_size // grid

    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    color_letters = ["R", "G", "B", "Y"]
    if len(digits) <= len(color_letters):
        chosen_colors = random.sample(color_letters, len(digits))
    else:
        chosen_colors = [random.choice(color_letters) for _ in digits]

    digit_to_color: Dict[str, str] = {d: c for d, c in zip(digits, chosen_colors)}

    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    positions = corners[:]

    special_digit = digits[0]

    base_font_size = int(img_size * 0.20)
    special_font_mult = 1.0
    apply_occlusion = False
    apply_blur = False
    non_special_desat = 0.0

    if condition == "sym":
        pass
    elif condition == "center":
        remaining = corners[:]
        random.shuffle(remaining)
        positions = [(1, 1)] + remaining
        positions = positions[: len(digits)]
    elif condition == "size":
        special_font_mult = 1.6
    elif condition == "contrast":
        non_special_desat = 0.55
    elif condition == "occlusion":
        apply_occlusion = True
    elif condition == "blur":
        apply_blur = True
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if condition != "center":
        random.shuffle(positions)
        positions = positions[: len(digits)]

    placement: Dict[str, Tuple[int, int]] = {d: pos for d, pos in zip(digits, positions)}

    for r in range(grid):
        for c in range(grid):
            x0 = c * cell
            y0 = r * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), outline=(210, 210, 210), width=1)

    for d in digits:
        r, c = placement[d]
        x0 = c * cell
        y0 = r * cell
        x1 = x0 + cell
        y1 = y0 + cell

        is_special = d == special_digit
        font_size = int(base_font_size * (special_font_mult if is_special else 1.0))
        font = load_font(font_size)

        col_letter = digit_to_color[d]
        base_rgb = COLOR_LETTER_TO_RGB[col_letter]

        if condition == "contrast" and not is_special:
            bg_rgb = blend_rgb(base_rgb, gray_level=160, alpha=non_special_desat)
        else:
            bg_rgb = base_rgb

        draw_digit_in_cell(draw, (x0, y0, x1, y1), d, font=font, bg_color=bg_rgb)

        if is_special and apply_occlusion:
            ox0 = x0 + int(cell * 0.12)
            ox1 = x1 - int(cell * 0.12)
            oy0 = y0 + int(cell * 0.45)
            oy1 = y0 + int(cell * 0.62)
            draw.rectangle((ox0, oy0, ox1, oy1), fill=(185, 185, 185), outline=None)

    if apply_blur:
        r, c = placement[special_digit]
        x0 = c * cell
        y0 = r * cell
        x1 = x0 + cell
        y1 = y0 + cell
        patch = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=2.0))
        img.paste(patch, (x0, y0))

    meta = {
        "condition": condition,
        "digits": digits,
        "special_digit": special_digit,
        "placement": {d: {"row": placement[d][0], "col": placement[d][1]} for d in digits},
        "digit_to_color": digit_to_color,
        "params": {
            "img_size": img_size,
            "grid": grid,
            "base_font_size": base_font_size,
            "special_font_mult": special_font_mult,
            "non_special_desat": non_special_desat,
            "occlusion": apply_occlusion,
            "blur": apply_blur,
        },
    }

    return img, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="poc_data_attr")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--per_condition", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--conditions",
        type=str,
        default="sym,center,size,contrast,occlusion,blur",
        help="Comma-separated list",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    pool = [str(i) for i in range(10)]

    idx = 0
    with open(meta_path, "w", encoding="utf-8") as handle:
        for cond in conditions:
            for _ in range(args.per_condition):
                digits = random.sample(pool, args.k)
                img, meta = make_image_and_meta(
                    img_size=args.img_size,
                    condition=cond,
                    digits=digits,
                    seed=args.seed * 100000 + idx,
                )
                img_name = f"{cond}_{idx:05d}.png"
                img_path = img_dir / img_name
                img.save(img_path)

                record = {"id": idx, "image_path": str(img_path), **meta}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                idx += 1

    print(f"[OK] Wrote {idx} images")
    print(f"[OK] Metadata: {meta_path}")


if __name__ == "__main__":
    main()
