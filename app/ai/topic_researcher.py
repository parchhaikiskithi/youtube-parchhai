from __future__ import annotations

import logging
from typing import Any

from app.ai.providers import LLMRouter
from app.database import Repository, session_scope

logger = logging.getLogger(__name__)

GENRES = [
    "horror",
    "mystery",
    "suspense",
    "paranormal",
    "haunted_haveli",
    "witch_ghost",
    "mysterious_village",
    "locked_room",
    "missing_person",
    "mysterious_letters",
    "shocking_twist",
]


class TopicResearcher:
    """Suggests original horror and mystery topics for 'परछाई किसकी थी?' using channel lessons."""

    def __init__(self, llm: LLMRouter | None = None) -> None:
        self.llm = llm or LLMRouter()

    def suggest_topic(
        self,
        language: str,
        genre: str,
        is_short: bool,
        avoid_topics: list[str] | None = None,
    ) -> dict[str, Any]:
        lessons: list[str] = []
        with session_scope() as session:
            for row in Repository(session).get_recent_lessons(15):
                lessons.append(f"[{row.category}] {row.lesson}")

        avoid = avoid_topics or []
        prompt = f"""You are the Master Horror & Mystery Story Director for YouTube Channel: "परछाई किसकी थी?".
Language: {language}
Genre: {genre}
Format: {"YouTube Short (under 60s narration)" if is_short else "Long-form Hindi horror/mystery (10-15 min narration)"}

STRICT FORBIDDEN CONTENT (NEVER PROPOSE):
- NO romantic love stories without a horror/mystery plot.
- NO normal family dramas or artificial tears.
- NO motivational stories or moral lectures.
- NO comedy stories.
- NO ordinary village life without supernatural/mystery element.
- NO random mythology or unrelated crime stories.
- NO predictable AI clichés or repeated ghost tropes.

ALLOWED CATEGORIES (CHOOSE ONLY FROM THESE):
1. Haunted Haveli (भूतिया हवेली)
2. Witch & Ghost Stories (चुड़ैल और आत्मा)
3. Mysterious Village (रहस्यमयी गाँव)
4. Locked Room Mystery (बंद कमरे का रहस्य)
5. Missing Person Mystery (गायब इंसान)
6. Mysterious Letters & Messages (रहस्यमयी चिट्ठी)
7. Shocking Twist Horror Stories (चौंकाने वाला मोड़)
8. Paranormal Fiction (अलौकिक घटनाएँ)

CONCEPT EVALUATION CRITERIA:
1. Genuine Fear & Atmosphere (psychological dread, uncertainty, dark setting)
2. Cold Hook (First 30 seconds terrifying incident or impossible discovery)
3. Unanswered Mystery (clues planted, rising suspense every 30-45s)
4. Shocking Logical Twist (unexpected revelation connecting to earlier clues)
5. High Click-Through Thumbnail Concept (3-6 word Hindi headline, intense focal subject)

Past lessons from this channel (apply them):
{chr(10).join(lessons) if lessons else "None yet."}

Topics to avoid (already used recently):
{", ".join(avoid) if avoid else "None specified."}

Respond with JSON only:
{{
  "main_topic": "खौफ़नाक और रहस्यमयी हिंदी शीर्षक (Devanagari)",
  "audience_hook": "पहले 30 सेकंड का रोंगटे खड़े कर देने वाला दृश्य या रहस्यमयी संवाद",
  "conflict_and_stakes": "मुख्य पात्र किस भयानक रहस्य या खतरे में फँसा है",
  "planned_twist": "कहानी का अप्रत्याशित और चौंकाने वाला मोड़ जो पहले के सुरागों से जुड़ा हो",
  "search_intent": "दर्शक इस रहस्य या भूतिया विषय को क्यों देखना चाहते हैं",
  "content_gap": "यह आम यूट्यूब कहानियों से कैसे अलग और खौफनाक है",
  "suggested_keywords": ["परछाई किसकी थी", "hindi horror story", "haunted haveli", "sachi ghatna", "rahasya"],
  "thumbnail_idea": "Ultra-realistic horror visual moment with single clear focal subject",
  "thumbnail_headline": "3-6 words punchy Hindi curiosity headline (Devanagari)",
  "uniqueness_note": "Why this horror/mystery concept is original, unpredictable and terrifying"
}}"""

        try:
            response = self.llm.complete(
                prompt,
                system=(
                    "You are the dedicated Horror & Mystery Story Director for 'परछाई किसकी थी?'. "
                    "Output valid JSON only. Exclusively generate terrifying, unpredictable, high-suspense concepts."
                ),
                json_mode=True,
            )
            return self.llm.parse_json(response.text)
        except Exception as exc:
            logger.warning("Topic research LLM failed: %s. Using procedural horror topic.", exc)
            return {
                "main_topic": "हवेली का रहस्य: परछाई किसकी थी?",
                "audience_hook": "रात के ठीक बारह बजे, बंद कमरे के आईने में एक अजनबी परछाई मुस्कुरा रही थी...",
                "conflict_and_stakes": "हवेली का सच जानने के चक्कर में राघव की अपनी जान खतरे में पड़ जाती है।",
                "planned_twist": "जिस परछाई से वह बच रहा था, वह दरअसल उसके अपने अतीत का एक खौफ़नाक सच थी।",
                "search_intent": "सच्ची भूतिया घटना और रोंगटे खड़े कर देने वाला रहस्य",
                "content_gap": "Atmospheric psychological dread with no cheap jump scares",
                "suggested_keywords": ["परछाई किसकी थी", "hindi horror stories", "haunted haveli", "bhoot ki kahani", "rahasya"],
                "thumbnail_idea": "A terrified young man looking into an antique mirror where a dark silhouette stands behind him, dim lantern light, misty dark haveli",
                "thumbnail_headline": "परछाई किसकी थी?",
                "uniqueness_note": "A deep psychological haunted haveli mystery with planted clues and a shocking mirror revelation",
            }
