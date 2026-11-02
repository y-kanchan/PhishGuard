"""Draw the PhishGuard shield icon at the sizes Chrome needs (extension/icons/)."""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "extension" / "icons"


def shield(size):
    s = 4  # supersample for smooth edges
    n = size * s
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = n * 0.08
    top, bottom, left, right = pad, n - pad, pad * 1.4, n - pad * 1.4
    mid = n / 2
    outline = [(mid, top), (right, top + n * 0.14), (right, n * 0.52), (mid, bottom), (left, n * 0.52), (left, top + n * 0.14)]
    d.polygon(outline, fill=(75, 99, 121, 255))  # Pantone Blue Fusion
    # check mark
    w = max(int(n * 0.09), 1)
    d.line([(n * 0.33, n * 0.5), (n * 0.46, n * 0.63), (n * 0.69, n * 0.36)], fill=(240, 238, 233), width=w, joint="curve")
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 48, 128):
        shield(size).save(OUT / f"icon{size}.png")
    print(f"icons -> {OUT}")
