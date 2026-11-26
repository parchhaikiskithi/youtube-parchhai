from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from app.content.categories import get_category

YOUTUBE_THUMB_SIZE = (1280, 720)

_CATEGORY_GRADIENTS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    # 8 Allowed Horror & Mystery Categories for "परछाई किसकी थी?"
    "haunted_haveli": ((10, 8, 25), (90, 15, 30)),
    "witch_ghost": ((15, 20, 30), (60, 20, 80)),
    "mysterious_village": ((15, 25, 20), (70, 45, 25)),
    "locked_room_mystery": ((25, 10, 15), (100, 30, 20)),
    "missing_person": ((10, 20, 35), (40, 60, 80)),
    "mysterious_letters": ((20, 15, 10), (110, 50, 30)),
    "shocking_twist_horror": ((5, 5, 20), (120, 20, 20)),
    "paranormal_fiction": ((10, 10, 40), (40, 90, 120)),
    # Legacy fallbacks
    "horror_mystery": ((10, 10, 30), (80, 20, 40)),
    "moral": ((10, 8, 25), (90, 15, 30)),
    "village_life": ((15, 25, 20), (70, 45, 25)),
    "suspense_thriller": ((5, 5, 20), (120, 20, 20)),
    "emotional": ((90, 50, 70), (255, 180, 140)),
    "motivational": ((20, 60, 100), (255, 140, 50)),
    "royal": ((60, 20, 20), (220, 180, 60)),
    "kids": ((40, 120, 180), (255, 220, 80)),
    "adventure": ((20, 40, 90), (60, 180, 200)),
    "mythology_inspired": ((40, 25, 90), (200, 120, 40)),
    "folk_mythology": ((60, 35, 20), (180, 140, 60)),
}


def _gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    w, h = size
    im = Image.new("RGB", size, top)
    draw = ImageDraw.Draw(im)
    for y in range(h):
        t = y / max(h - 1, 1)
        color = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line([(0, y), (w, y)], fill=color)
    return im


def create_story_fallback_art(category_id: str, out: Path) -> Path:
    """Abstract category poster when SDXL is unavailable — never a video frame."""
    get_category(category_id)  # validate id
    colors = _CATEGORY_GRADIENTS.get(category_id, ((30, 40, 80), (140, 90, 50)))
    im = _gradient(YOUTUBE_THUMB_SIZE, colors[0], colors[1])
    draw = ImageDraw.Draw(im)
    cx, cy = 640, 320
    for r in range(260, 60, -30):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 220, 150))
    im = im.filter(ImageFilter.GaussianBlur(radius=1.0))
    im = ImageEnhance.Contrast(im).enhance(1.18)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, format="PNG", optimize=True)
    return out
