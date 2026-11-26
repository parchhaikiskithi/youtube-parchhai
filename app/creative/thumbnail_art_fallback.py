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
    """Atmospheric dark horror silhouette poster when AI image gen is unavailable — never geometric circles."""
    get_category(category_id)  # validate id
    colors = _CATEGORY_GRADIENTS.get(category_id, ((10, 8, 25), (70, 15, 25)))
    im = _gradient(YOUTUBE_THUMB_SIZE, colors[0], colors[1])
    draw = ImageDraw.Draw(im, "RGBA")
    w, h = YOUTUBE_THUMB_SIZE

    # Atmospheric moonlight glow in upper center/left
    for r in range(180, 20, -20):
        alpha = int(45 * (1 - r / 180))
        draw.ellipse([340 - r, 200 - r, 340 + r, 200 + r], fill=(180, 210, 255, alpha))
    # Full moon disc
    draw.ellipse([340 - 45, 200 - 45, 340 + 45, 200 + 45], fill=(230, 240, 255, 220))

    # Dark silhouette of ancient haunted mansion / haveli structure in background
    haveli_pts = [
        (0, h), (0, 380), (120, 380), (140, 310), (180, 310), (200, 380),
        (380, 380), (410, 260), (450, 220), (490, 260), (520, 380),
        (700, 380), (740, 330), (780, 330), (820, 380), (1050, 380),
        (1080, 280), (1140, 280), (1170, 380), (w, 380), (w, h)
    ]
    draw.polygon(haveli_pts, fill=(8, 7, 15, 240))

    # Glowing eerie red / amber windows in the dark mansion
    window_glows = [
        (435, 280, 465, 330), (150, 330, 170, 360), (750, 345, 770, 370), (1100, 305, 1120, 345)
    ]
    for wx0, wy0, wx1, wy1 in window_glows:
        draw.rounded_rectangle([wx0, wy0, wx1, wy1], radius=8, fill=(240, 50, 20, 210))
        # Inner warm slit
        draw.rectangle([wx0 + 4, wy0 + 4, wx1 - 4, wy1 - 4], fill=(255, 180, 60, 230))

    # Creepy barren tree branches silhouette
    tree_branches = [
        [(150, h), (180, 450), (120, 350), (90, 280)],
        [(180, 450), (230, 370), (270, 310)],
        [(1150, h), (1110, 430), (1160, 320), (1220, 260)],
        [(1110, 430), (1060, 350), (1020, 290)]
    ]
    for branch in tree_branches:
        draw.line(branch, fill=(5, 4, 10, 230), width=6)

    # Menacing shadowy ghost figure silhouette lurking in foreground right
    ghost_pts = [
        (850, h), (860, 420), (880, 360), (910, 320), (940, 320),
        (970, 360), (990, 420), (1000, h)
    ]
    draw.polygon(ghost_pts, fill=(2, 2, 6, 250))
    # Glowing piercing crimson eyes of the shadowy figure
    draw.ellipse([900, 350, 915, 360], fill=(255, 20, 20, 255))
    draw.ellipse([935, 350, 950, 360], fill=(255, 20, 20, 255))

    # Dense atmospheric ground fog / mist layers
    for y_fog in range(h - 180, h, 20):
        alpha_fog = int(60 * ((y_fog - (h - 180)) / 180))
        draw.rectangle([0, y_fog, w, y_fog + 20], fill=(20, 30, 50, alpha_fog))

    # Heavy dark cinematic vignette around borders
    vignette = Image.new("RGBA", YOUTUBE_THUMB_SIZE, (0, 0, 0, 0))
    vdraw = ImageDraw.Draw(vignette)
    for i in range(120):
        alpha = int(180 * (1 - i / 120))
        vdraw.rectangle([i, i, w - i, h - i], outline=(0, 0, 0, alpha))
    im = Image.alpha_composite(im.convert("RGBA"), vignette).convert("RGB")

    im = im.filter(ImageFilter.GaussianBlur(radius=0.8))
    im = ImageEnhance.Contrast(im).enhance(1.22)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, format="PNG", optimize=True)
    return out
