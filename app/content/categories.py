from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContentCategory:
    id: str
    name_hi: str
    name_en: str
    legacy_genre: str
    default_weight: float
    audience: str
    story_style: str
    visual_style: str
    voice_profile: str
    audio_profile: str
    thumbnail_tone: str
    kids_safe: bool = False


# The 8 Allowed Horror & Mystery Categories for YouTube Channel: "परछाई किसकी थी?"
CATEGORIES: dict[str, ContentCategory] = {
    "haunted_haveli": ContentCategory(
        id="haunted_haveli",
        name_hi="भूतिया हवेली",
        name_en="Haunted Haveli",
        legacy_genre="horror",
        default_weight=0.15,
        audience="adult_teens",
        story_style=(
            "Years-old abandoned mansions, haunted royal palaces, mysterious locked rooms inside old havelis, "
            "families who disappeared from mansions, secret underground tunnels, strange sounds coming from locked rooms, "
            "mansions where nobody survives after midnight, hidden histories and supernatural secrets."
        ),
        visual_style="ancient abandoned Indian royal haveli, dark storm, cracked pillars, dim flickering lantern, fog, moody shadows, photorealistic horror cinema 8k",
        voice_profile="horror",
        audio_profile="horror",
        thumbnail_tone="haunted mansion, crimson light, terror",
    ),
    "witch_ghost": ContentCategory(
        id="witch_ghost",
        name_hi="चुड़ैल और आत्मा",
        name_en="Witch & Ghost Stories",
        legacy_genre="horror",
        default_weight=0.15,
        audience="adult_teens",
        story_style=(
            "Mysterious witches, spirits of abandoned villages, ghosts appearing at midnight, spirits seeking revenge, "
            "mysterious women seen on lonely roads, ghosts with hidden identities, spirits connected to forgotten incidents, "
            "supernatural encounters."
        ),
        visual_style="lonely dark Indian road, eerie mist, mysterious pale silhouette, glowing eyes in darkness, cinematic horror atmosphere",
        voice_profile="horror",
        audio_profile="horror",
        thumbnail_tone="eerie apparition, terrified face, dark fog",
    ),
    "mysterious_village": ContentCategory(
        id="mysterious_village",
        name_hi="रहस्यमयी गाँव",
        name_en="Mysterious Village",
        legacy_genre="mystery",
        default_weight=0.15,
        audience="adult_teens",
        story_style=(
            "Villages where nobody goes outside after sunset, villages disappearing from maps, houses where strange incidents happen, "
            "a village with a forbidden forest, mysterious village traditions, people disappearing every full moon, "
            "a village hiding a terrible secret, a village where nobody speaks about the past."
        ),
        visual_style="isolated Indian village at twilight, deserted dusty streets, closed wooden shutters, creeping shadows, eerie mist",
        voice_profile="horror",
        audio_profile="thriller",
        thumbnail_tone="deserted village, shadowy figure, fear",
    ),
    "locked_room_mystery": ContentCategory(
        id="locked_room_mystery",
        name_hi="बंद कमरे का रहस्य",
        name_en="Locked Room Mystery",
        legacy_genre="mystery",
        default_weight=0.10,
        audience="adult_teens",
        story_style=(
            "A locked room with no visible entrance, someone disappearing from a locked room, mysterious deaths inside closed rooms, "
            "strange messages appearing on walls, a room that opens only at midnight, sounds coming from an empty room, "
            "hidden passages and secret doors, impossible mysteries with logical or supernatural explanations."
        ),
        visual_style="heavy chained rustic wooden door, scratches on ancient walls, candle illumination, blood-red occult symbols, high contrast shadow",
        voice_profile="horror",
        audio_profile="thriller",
        thumbnail_tone="locked chained door, shock expression, dark mystery",
    ),
    "missing_person": ContentCategory(
        id="missing_person",
        name_hi="गायब इंसान",
        name_en="Missing Person Mystery",
        legacy_genre="suspense",
        default_weight=0.10,
        audience="adult_teens",
        story_style=(
            "People disappearing without evidence, a missing child returning years later with eerie secrets, "
            "someone disappearing from a moving vehicle, a person seen by multiple witnesses but never found, "
            "families receiving messages from missing people, people disappearing from isolated houses, an identity mystery."
        ),
        visual_style="faded missing person poster under flickering streetlamp, rain-soaked night, empty dark vehicle, haunting silhouette",
        voice_profile="horror",
        audio_profile="thriller",
        thumbnail_tone="missing person poster, dread, revelation",
    ),
    "mysterious_letters": ContentCategory(
        id="mysterious_letters",
        name_hi="रहस्यमयी चिट्ठी",
        name_en="Mysterious Letters & Messages",
        legacy_genre="mystery",
        default_weight=0.10,
        audience="adult_teens",
        story_style=(
            "Letters arriving from dead people, anonymous letters predicting future incidents, messages warning people not to enter a place, "
            "letters revealing forgotten secrets, old diaries containing mysterious information, messages appearing from unknown numbers, "
            "a letter that changes the entire story."
        ),
        visual_style="yellowed aged parchment letter written in red ink, vintage fountain pen, trembling hands, dim candlelight, ominous atmosphere",
        voice_profile="horror",
        audio_profile="thriller",
        thumbnail_tone="blood stained letter, horrified eyes, candle",
    ),
    "shocking_twist_horror": ContentCategory(
        id="shocking_twist_horror",
        name_hi="चौंकाने वाला मोड़",
        name_en="Shocking Twist Horror Stories",
        legacy_genre="suspense",
        default_weight=0.15,
        audience="adult_teens",
        story_style=(
            "Stories with unexpected final revelations, the trusted person turns out to be the mystery, the main character discovers an unbelievable truth, "
            "a ghost is connected to the protagonist's past, the entire story has a hidden second meaning, the final scene completely changes understanding, "
            "endings that are surprising but logically consistent with earlier clues."
        ),
        visual_style="fractured mirror reflection showing double terrifying face, dark rainy backdrop, intense psychological tension, cinematic noir",
        voice_profile="horror",
        audio_profile="thriller",
        thumbnail_tone="extreme shock, twist reveal, shattered reflection",
    ),
    "paranormal_fiction": ContentCategory(
        id="paranormal_fiction",
        name_hi="अलौकिक घटनाएँ",
        name_en="Paranormal Fiction",
        legacy_genre="horror",
        default_weight=0.10,
        audience="adult_teens",
        story_style=(
            "Supernatural activities, haunted objects, mysterious shadows moving independently, unexplained voices, strange reflections, "
            "objects moving without human intervention, time-related supernatural mysteries, dreams connecting to real incidents, "
            "unexplained events with terrifying consequences."
        ),
        visual_style="mysterious levitating antique object, unnatural elongated black shadow on wall, cold blue moonlight, chilling paranormal presence",
        voice_profile="horror",
        audio_profile="horror",
        thumbnail_tone="living shadow, creeping horror, supernatural glow",
    ),
}

# Legacy category aliases redirected to appropriate Horror & Mystery categories
LEGACY_ALIASES: dict[str, str] = {
    "moral": "haunted_haveli",
    "horror_mystery": "haunted_haveli",
    "village_life": "mysterious_village",
    "suspense_thriller": "shocking_twist_horror",
    "emotional_family": "shocking_twist_horror",
    "inspirational": "paranormal_fiction",
    "raja_rani_fantasy": "haunted_haveli",
    "kids_animal": "paranormal_fiction",
    "folk_mythology": "witch_ghost",
    "fantasy_adventure": "paranormal_fiction",
    "life_lessons": "missing_person",
    "indian_folk": "witch_ghost",
    "mythology_inspired": "witch_ghost",
}


def get_category(category_id: str) -> ContentCategory:
    if category_id in CATEGORIES:
        return CATEGORIES[category_id]
    alias = LEGACY_ALIASES.get(category_id)
    if alias and alias in CATEGORIES:
        return CATEGORIES[alias]
    return CATEGORIES["haunted_haveli"]


def all_category_ids() -> list[str]:
    return list(CATEGORIES.keys())


def category_story_block(category: ContentCategory) -> str:
    return f"""
YouTube Channel: परछाई किसकी थी?
Category: {category.name_hi} ({category.name_en})
Audience: {category.audience}
Category Story Directives:
{category.story_style}
Visual Style Hint: {category.visual_style}

HIGH RETENTION HORROR & SUSPENSE DIRECTIVES:
- Hook in first 30 seconds (begin with the most mysterious, terrifying or impossible incident; NEVER start with boring intro).
- Pacing: Introduce a new development, clue, or escalate fear every 30-45 seconds.
- Realistic conversational Hindi dialogues with emotional pauses and character-specific voices.
- 8-Part Cinematic Structure: Hook -> Character Intro -> First Strange Incident -> Suspense Development -> Major Revelation -> Climax -> Shocking Twist -> Meaningful Ending.
- Clues planted early; final revelation must surprise viewers and make logical sense in retrospect.
"""


def format_duration_range(mode: str) -> tuple[int, int, int, int]:
    """Returns min_sec, max_sec, min_scenes, max_scenes."""
    if mode == "short":
        return 30, 60, 4, 6
    if mode == "extended":
        return 720, 1080, 18, 25
    if mode == "compilation":
        return 900, 1500, 20, 28
    # Default 10–15 minutes (600–900 seconds) for complete, cinematic Hindi horror stories (15-22 scenes)
    return 600, 900, 15, 22
