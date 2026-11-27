from __future__ import annotations

import json
import logging
from typing import Any

from app.ai.providers import LLMRouter
from app.ai.repetition_detector import RepetitionDetector
from app.ai.story_blueprint_generator import StoryBlueprint

logger = logging.getLogger(__name__)


class SceneBasedStoryWriter:
    """
    Writes rich, scene-by-scene cinematic Hindi stories strictly following the SHOW, DON'T TELL mandate.
    Every important event is structured with Setup, concrete Event, physical Action,
    meaningful Dialogue, character Reaction, and immediate Consequence.
    """

    def __init__(self, llm: LLMRouter | None = None) -> None:
        self.llm = llm or LLMRouter()
        self.repetition_detector = RepetitionDetector()

    @staticmethod
    def build_system_prompt() -> str:
        return (
            "# MASTER SYSTEM PROMPT: CINEMATIC HINDI SCENE-BASED STORY WRITER\n"
            "## Channel: परछाई किसकी थी?\n\n"
            "You are a master Bollywood Screenwriter, Hindi Novelist, and Story Director.\n"
            "Your writing must follow a STRICT 'SHOW, DON'T TELL' approach.\n\n"
            "### 1. ABSOLUTE BAN ON VAGUE STORYTELLING:\n"
            "NEVER summarize important problems or developments with empty phrases like:\n"
            "- 'उसे जिंदगी में बहुत मुश्किलों का सामना करना पड़ा।'\n"
            "- 'उसकी परेशानियां बढ़ती चली गईं।'\n"
            "- 'हालात बहुत खराब हो गए।'\n"
            "- 'परिवार पर दुखों का पहाड़ टूट पड़ा।'\n"
            "- 'उसकी जिंदगी में तूफान आ गया।'\n"
            "- 'उसने बहुत कुछ सहा।'\n"
            "- 'सब कुछ बदल गया।'\n\n"
            "Whenever the story mentions a problem, betrayal, conflict, discovery, crime, or mystery, "
            "YOU MUST EXPLAIN THE ACTUAL EVENT THROUGH DETAILED SCENES:\n"
            "- What exactly happened?\n"
            "- Who caused the problem?\n"
            "- Where and when did it happen?\n"
            "- What did the character see and hear?\n"
            "- What did the characters say?\n"
            "- How did they physically react?\n"
            "- What were the tangible consequences?\n"
            "- What happened next?\n\n"
            "### 2. COMPLETE SCENE STRUCTURE:\n"
            "Every scene must have:\n"
            "A. SETUP: Where is the character? What is happening?\n"
            "B. EVENT: What concrete incident occurs?\n"
            "C. ACTION: What do the characters physically do?\n"
            "D. DIALOGUE: Meaningful conversations that reveal clues, relationships, or conflict.\n"
            "E. REACTION: Realistic emotional and physical response.\n"
            "F. CONSEQUENCE: What changes because of this event?\n\n"
            "### 3. ABSOLUTE BAN ON REPETITION & LEXICAL MONOTONY:\n"
            "- NEVER repeat the same words or crutch phrases across consecutive sentences or scenes (e.g. 'को लगता है कि', 'खड़ा होता है', 'देखता है कि', 'अजीब', 'भयानक', 'सच्चाई').\n"
            "- Use rich, varied Hindi vocabulary, evocative synonyms, and dynamic sentence structures.\n"
            "- NEVER repeat the same information using different words across scenes.\n"
            "- NEVER reuse questions like 'यह क्या है?', 'तुम कौन हो?', 'यह क्या हो रहा है?'.\n"
            "- Every paragraph must contribute at least ONE NEW THING: new clue, new action, new conflict, or new revelation.\n\n"
            "### 4. SATISFYING RESOLUTION:\n"
            "- Never finish with clichés like 'और इस तरह जिंदगी बदल गई' or 'समय के साथ सब ठीक हो गया'.\n"
            "- Show what happened to each character, how the central mystery was solved, and the final aftermath.\n\n"
            "Output valid JSON ONLY."
        )

    def write_story_from_blueprint(
        self,
        blueprint: StoryBlueprint,
        category_name: str,
        target_scenes: int = 15,
        target_duration_sec: int = 750,
        visual_style: str = "Cinematic 35mm photo, ultra-realistic horror cinema, 8k",
    ) -> dict[str, Any]:
        """Generates the full story JSON based on the pre-constructed blueprint."""
        prompt = f"""Write a masterclass cinematic Hindi horror/mystery story script based on this detailed plot blueprint.

PLOT BLUEPRINT:
{json.dumps(blueprint.to_dict(), ensure_ascii=False, indent=2)}

TARGET SCENES: Exactly {target_scenes} scenes.
Target spoken duration: ~{target_duration_sec} seconds.
Visual Style: {visual_style}

CRITICAL EXECUTION REQUIREMENTS:
1. SCENE NARATION DEPTH: Every scene MUST have 4-7 rich sentences of atmospheric narration (minimum 320 Devanagari characters) with concrete actions, sensory details, and tangible events.
2. DIALOGUE QUALITY: Every scene MUST have 2-4 lines of natural conversational Hindi dialogue between characters that creates tension, reveals information, or escalates conflict.
3. SEPARATE AUDIO: Narration sets the scene context; dialogue carries character speech. Never quote spoken dialogue inside narration.
4. STRICT SHOW, DON'T TELL: No empty assertions. Show the actual events!
5. ZERO REPETITION: Every scene must advance the narrative forward.

Return JSON in this exact structure:
{{
  "title": "{blueprint.concept[:60]}",
  "synopsis": "{blueprint.concept}",
  "characters": [ ... ],
  "character_references": [ ... ],
  "scenes": [
    {{
      "scene_number": 1,
      "story_beat": "opening_hook",
      "visual_description": "Cinematic photo...",
      "image_prompt": "Cinematic photo...",
      "narration": "Devanagari narrative paragraph with concrete actions and sensory details...",
      "dialogue": [
        {{"character": "name", "line": "Meaningful Hindi dialogue", "gender": "male|female", "acting_instruction": "emotion"}}
      ],
      "mood": "horror",
      "characters_present": ["name"]
    }}
  ],
  "hook_opening": "First scene hook...",
  "ending": "{blueprint.final_resolution}",
  "conclusion_moral": "{blueprint.conclusion_moral}",
  "call_to_action": "अगर आपको यह खौफ़नाक कहानी पसंद आई, तो वीडियो को लाइक करें और चैनल को सब्सक्राइब करें।"
}}"""

        try:
            resp = self.llm.complete(
                prompt,
                system=self.build_system_prompt(),
                json_mode=True,
            )
            data = self.llm.parse_json(resp.text)
            if data and data.get("scenes") and len(data["scenes"]) >= 4:
                return self.repetition_detector.deduplicate_story(data)
        except Exception as exc:
            logger.warning("LLM scene writing from blueprint failed (%s)", exc)

        return {}
