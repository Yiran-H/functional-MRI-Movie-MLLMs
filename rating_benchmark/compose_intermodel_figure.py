"""
compose_intermodel_figure.py — Composite the inter-model comparison figures
into one publication panel (the correlation-matrix figure,
fig_model_agreement_matrix.png, stands alone as its own separate figure and
is NOT part of this composite). Layout: single row, left-to-right --
    Left:  fig_rating_distribution.png   (rating distribution, human vs models)
    Right: fig_tom_valence_consistency.png (ToM vs Valence consistency scatter)
fig_agreement_summary.png (the old panel b) is no longer part of this
composite.

Usage:
    python compose_intermodel_figure.py
Output:
    results/9.7/intermodel/fig_intermodel_composite.png (+.pdf)
"""
import os
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
# Source images come straight from results/ -- where rating_distribution_calibration.py
# and tom_valence_consistency_scatter.py actually write them -- NOT the
# results/9.7/intermodel/ stale-copy folder (which only holds old snapshots
# and was silently going out of sync with re-runs of those scripts).
SRC_DIR = os.path.join(_HERE, "results")
OUT_DIR = os.path.join(_HERE, "results", "9.7", "intermodel")

LEFT = "fig_rating_distribution.png"
RIGHT = "fig_tom_valence_consistency.png"

PAD = 24
TOP_PAD = 240  # extra headroom above the panels so a)/b) labels have real white space to sit in
GAP = 60  # gap between left and right panels
LABEL_FONT = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
LABEL_SIZE = 350
LABEL_MARGIN = -20  # horizontal inset from each panel's left edge


def scaled_to_height(img, target_h):
    w, h = img.size
    scale = target_h / h
    return img.resize((int(w * scale), target_h), Image.LANCZOS)


def main():
    left_img = Image.open(os.path.join(SRC_DIR, LEFT)).convert("RGB")
    right_img = Image.open(os.path.join(SRC_DIR, RIGHT)).convert("RGB")

    # match heights (use the taller of the two as the common row height)
    row_h = max(left_img.height, right_img.height)
    left_img = scaled_to_height(left_img, row_h)
    right_img = scaled_to_height(right_img, row_h)

    canvas_w = left_img.width + GAP + right_img.width + PAD * 2
    canvas_h = TOP_PAD + row_h + PAD

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(left_img, (PAD, TOP_PAD))
    canvas.paste(right_img, (PAD + left_img.width + GAP, TOP_PAD))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(LABEL_FONT, LABEL_SIZE)
    label_y = (TOP_PAD - LABEL_SIZE) // 2
    draw.text((PAD + LABEL_MARGIN, label_y), "a)", fill="black", font=font)
    draw.text((PAD + left_img.width + GAP + LABEL_MARGIN, label_y), "b)", fill="black", font=font)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "fig_intermodel_composite.png")
    canvas.save(out_png, dpi=(600, 600))
    canvas.save(out_png.replace(".png", ".pdf"))
    print(f"Saved -> {out_png} (+.pdf)  size={canvas.size}")


if __name__ == "__main__":
    main()
