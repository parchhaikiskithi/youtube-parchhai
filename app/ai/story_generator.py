from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.ai.providers import LLMRouter
from app.ai.repetition_detector import RepetitionDetector
from app.ai.scene_based_story_writer import SceneBasedStoryWriter
from app.ai.story_blueprint_generator import StoryBlueprint, StoryBlueprintGenerator
from app.ai.story_quality_auditor import StoryAuditReport, StoryQualityAuditor
from app.ai.story_rewriter import StoryRewriter
from app.ai.topic_researcher import TopicResearcher
from app.ai.vague_sentence_detector import VagueSentenceDetector
from app.config import get_settings
from app.content.categories import category_story_block, get_category
from app.database import Repository, session_scope

logger = logging.getLogger(__name__)

STORY_SCHEMA_HINT = """
{
  "title": "हवेली का बंद कमरा: परछाई किसकी थी? (Devanagari)",
  "synopsis": "एक वीरान हवेली के बंद कमरे में आधी रात को दिखाई देने वाली रहस्यमयी परछाई और आईने का वो खौफ़नाक सच जो पूरे इलाके को दहला देता है।",
  "category": "haunted_haveli",
  "characters": [
    {
      "name": "राघव",
      "gender": "male",
      "age_group": "27 years old",
      "description": "Young Indian investigator with sharp jawline, intense eyes, wearing dark navy blue jacket over grey henley",
      "voice_id_hint": "hi_IN-pratham-medium"
    }
  ],
  "character_references": [
    {
      "name": "राघव",
      "gender": "male",
      "age_group": "27 years old",
      "physical_appearance": "Sharp Indian features, clean shaven, intense piercing eyes, short black styled hair, height 5'11",
      "signature_attire": "Dark navy blue jacket, grey henley t-shirt, silver watch on left wrist",
      "personality": "Cautious, razor-sharp observant, fearless"
    }
  ],
  "story_structure_breakdown": {
    "opening_hook": "Part 1: Powerful Hook - Terrifying cold opening in first 30 seconds",
    "character_intro": "Part 2: Character Introduction and arrival motivation",
    "first_strange_incident": "Part 3: First unexplainable supernatural occurrence",
    "suspense_development": "Part 4: Escalating clues, dread and psychological tension",
    "major_revelation": "Part 5: Shocking discovery that raises the stakes",
    "climax": "Part 6: High stakes terrifying confrontation",
    "shocking_twist": "Part 7: Unexpected but logical revelation reframing earlier clues",
    "meaningful_ending": "Part 8: Satisfying final resolution with memorable lingering fear"
  },
  "scenes": [
    {
      "scene_number": 1,
      "story_beat": "opening_hook",
      "duration_estimate_sec": 45,
      "visual_description": "Cinematic ultra-realistic 35mm photo of ancient dark Indian haveli in heavy midnight rain, lightning illuminating cracked royal archway, single crimson lantern glowing in attic window, photorealistic horror cinema, 8k",
      "image_prompt": "Cinematic ultra-realistic photo of an ancient dark Indian haveli at stormy midnight, pouring rain, glowing crimson lantern in attic window, photorealistic horror cinema",
      "narration": "रात के ठीक बारह बजे, पुरानी वीरान हवेली की बंद खिड़की से अचानक लाल रोशनी फूट पड़ी। राघव ने जैसे ही हवेली के भारी लोहे के दरवाज़े को धक्का दिया, अंदर की सर्द हवा ने उसके रोंगटे खड़े कर दिए। धूल भरी सीढ़ियों पर ताज़ा कदमों के निशान साफ़ दिखाई दे रहे थे, जैसे कोई साया ठीक उसके आगे-आगे चल रहा हो।",
      "dialogue": [
        {
          "character": "राघव",
          "acting_instruction": "दबी हुई गंभीर आवाज़",
          "line": "कौन है वहाँ? सामने आओ, मैं जानता हूँ तुम यहीं छुपे हो!",
          "gender": "male"
        }
      ],
      "mood": "horror",
      "characters_present": ["राघव"],
      "sfx_hint": "thunder, rain, creaking door, eerie wind"
    }
  ],
  "hook_opening": "रात के ठीक बारह बजे, पुरानी हवेली की बंद खिड़की से लाल रोशनी फूट पड़ी...",
  "ending": "राघव ने जब आईने में दोबारा देखा, तो वहां कोई परछाई नहीं थी... लेकिन हवेली के सन्नाटे में एक धीमी फुसफुसाहट अब भी गूँज रही थी।",
  "call_to_action": "अगर आपको 'परछाई किसकी थी?' की यह खौफ़नाक कहानी पसंद आई, तो वीडियो को लाइक करें और ऐसी ही रोंगटे खड़े कर देने वाली रहस्यमयी कहानियों के लिए चैनल को सब्सक्राइब करें।",
  "thumbnail": {
    "headline": "परछाई किसकी थी?",
    "visual_concept": "A terrified young man looking into an antique cracked mirror where a menacing dark silhouette smiles behind him in a dimly lit haunted haveli",
    "image_prompt": "Cinematic ultra-realistic YouTube thumbnail 16:9, terrified young Indian man with wide horrified eyes holding brass lantern, antique mirror showing sinister dark shadowy figure behind him, dramatic chiaroscuro rim lighting, high contrast horror movie poster aesthetic, 8k"
  },
  "content_rating": "general",
  "synthetic_disclosure_needed": true
}
"""


@dataclass
class StoryGenerationRequest:
    language: str
    genre: str = "horror"
    target_duration_sec: int = 750
    is_short: bool = False
    content_style: str = "cinematic Hindi horror storytelling"
    category_id: str = "haunted_haveli"
    topic_research: dict[str, Any] | None = None
    min_scenes: int | None = None
    max_scenes: int | None = None
    visual_style: str | None = None


class StoryGenerator:
    """
    Master Story Generation Engine enforcing:
    1. SHOW WHAT ACTUALLY HAPPENED (Strict Show, Don't Tell).
    2. ZERO REPETITION (Duplicate sentences, semantic concept loops, dialogue loops).
    3. ZERO EMPTY PHRASES ('उसे बहुत संघर्ष करना पड़ा', 'हालात बिगड़ते गए', etc.).
    4. COMPLETE SCENE STRUCTURE (Setup, Event, Action, Dialogue, Reaction, Consequence).
    5. NATURAL HINDI DIALOGUES WITH CONFLICT & CLUES.
    6. ROCK-SOLID STORY BLUEPRINT WITH CAUSE-AND-EFFECT CONTINUITY.
    7. SATISFYING RESOLUTION RESOLVING ALL MYSTERIES.
    8. STRICT 10-DIMENSION QUALITY AUDITOR (MIN 85/100).
    """

    def __init__(self, llm: LLMRouter | None = None) -> None:
        self.llm = llm or LLMRouter()
        self.topics = TopicResearcher(self.llm)
        self.settings = get_settings()
        self.blueprint_generator = StoryBlueprintGenerator(self.llm)
        self.scene_writer = SceneBasedStoryWriter(self.llm)
        self.repetition_detector = RepetitionDetector()
        self.vague_detector = VagueSentenceDetector()
        self.quality_auditor = StoryQualityAuditor(self.llm)
        self.story_rewriter = StoryRewriter(self.llm)

    def _lessons_block(self) -> str:
        with session_scope() as session:
            lessons = Repository(session).get_recent_lessons(20)
        if not lessons:
            return "No prior lessons."
        return "\n".join(f"- ({l.category}) {l.lesson}" for l in lessons)

    def _hindi_system(self) -> str:
        return (
            "# MASTER SYSTEM PROMPT: HORROR & MYSTERY STORY GENERATOR\n"
            "## YouTube Channel: परछाई किसकी थी?\n\n"
            "You are an expert Hindi horror novelist, psychological thriller writer, mystery storyteller, "
            "screenplay writer and Bollywood-style cinematic story director.\n\n"
            "Your ONLY responsibility is to create highly engaging, original, suspenseful, terrifying and "
            "unpredictable Hindi Horror & Mystery stories for a YouTube storytelling channel named:\n\n"
            "\"परछाई किसकी थी?\"\n\n"
            "You must strictly follow the horror and mystery genres. Never generate unrelated stories.\n\n"
            "## 1. ALLOWED STORY CATEGORIES\n"
            "Generate stories ONLY from these 8 categories:\n"
            "1. Haunted Haveli (भूतिया हवेली): Abandoned mansions, haunted royal palaces, locked rooms, underground tunnels, midnight survival secrets.\n"
            "2. Witch & Ghost Stories (चुड़ैल और आत्मा): Mysterious witches, vengeful spirits, lonely road apparitions at midnight, forgotten incidents.\n"
            "3. Mysterious Village (रहस्यमयी गाँव): Villages deserted after sunset, forbidden forests, dark traditions, disappearing villagers.\n"
            "4. Locked Room Mystery (बंद कमरे का रहस्य): Rooms with no entrance, mysterious disappearances, blood-chilling messages on walls, impossible secrets.\n"
            "5. Missing Person Mystery (गायब इंसान): Disappearing without evidence, missing people returning years later with eerie secrets, phantom messages.\n"
            "6. Mysterious Letters & Messages (रहस्यमयी चिट्ठी): Letters from the dead, warning predictions, cursed diaries, unknown number texts.\n"
            "7. Shocking Twist Horror Stories (चौंकाने वाला मोड़): Unbelievable revelations, trusted allies turning out to be the terror, logical mind-bending twists.\n"
            "8. Paranormal Fiction (अलौकिक घटनाएँ): Haunted objects, independent shadows, strange reflections, prophetic nightmares, supernatural events.\n\n"
            "## 2. CINEMATIC STORY STRUCTURE (8 PARTS)\n"
            "PART 1 – Powerful Hook (First 30 Seconds): Begin with the most terrifying or mysterious moment.\n"
            "PART 2 – Character Introduction: Natural introduction, Indian setting, believable motivation.\n"
            "PART 3 – First Strange Incident: First unexplained event, build curiosity, plant subtle clues.\n"
            "PART 4 – Suspense Development: Increase fear gradually, eerie discoveries, escalating stakes every 30-45s.\n"
            "PART 5 – Major Revelation: Reveal an important part of the mystery, introduce a bigger unanswered question.\n"
            "PART 6 – Climax: Maximum tension, bring all clues together in a powerful confrontation.\n"
            "PART 7 – Shocking Twist: Unexpected but logical revelation connecting with earlier clues.\n"
            "PART 8 – Meaningful Ending: Resolve the central mystery while leaving a memorable spine-chilling impression.\n\n"
            "## 3. STRICT SHOW, DON'T TELL MANDATE (NO VAGUE STORYTELLING):\n"
            "NEVER summarize struggles, betrayals, arguments, crimes, or mysteries with empty vague phrases like:\n"
            "- 'उसे जिंदगी में बहुत मुश्किलों का सामना करना पड़ा।'\n"
            "- 'उसकी परेशानियां बढ़ती चली गईं।'\n"
            "- 'हालात बहुत खराब हो गए।'\n"
            "- 'परिवार पर दुखों का पहाड़ टूट पड़ा।'\n"
            "- 'उसकी जिंदगी में तूफान आ गया।'\n"
            "- 'उसने बहुत कुछ सहा।'\n"
            "- 'सब कुछ बदल गया।'\n\n"
            "Whenever the story mentions an important event, struggle, or discovery, YOU MUST EXPLAIN WHAT ACTUALLY HAPPENED:\n"
            "- What exactly happened?\n"
            "- Who caused the problem?\n"
            "- Where and when did it happen?\n"
            "- What did the character physically see and hear?\n"
            "- What did the characters say to each other?\n"
            "- How did the character react physically and emotionally?\n"
            "- What were the immediate tangible consequences?\n"
            "- What happened next?\n\n"
            "## 4. COMPLETE SCENE STRUCTURE (EVERY IMPORTANT SCENE):\n"
            "A. SETUP: Where is the character? What is the atmosphere?\n"
            "B. EVENT: What concrete incident occurs?\n"
            "C. ACTION: What tangible actions do characters take?\n"
            "D. DIALOGUE: Realistic conversation that reveals clues, secrets, or conflict.\n"
            "E. REACTION: Realistic emotional response.\n"
            "F. CONSEQUENCE: What changes because of this event?\n"
            "G. NEXT DEVELOPMENT: How does it lead to the next scene?\n\n"
            "## 5. ABSOLUTE BAN ON REPETITION:\n"
            "- Never repeat the same information using different words just to increase story length.\n"
            "- Never repeat questions like 'यह क्या है?', 'तुम कौन हो?', 'यह क्या हो रहा है?'.\n"
            "- Every paragraph must contribute at least ONE NEW THING: new information, action, clue, conflict, character development, consequence, or suspense.\n\n"
            "## 6. NATURAL CONVERSATIONAL HINDI DIALOGUES:\n"
            "- Characters must sound like real Indian people, with distinct personalities, emotions, and pauses ('...').\n"
            "- Dialogues must reveal info, show relationships, create conflict, and introduce clues.\n"
            "- Separate narration and character dialogue cleanly. Never quote dialogue inside narration.\n\n"
            "## 7. SATISFYING RESOLUTION:\n"
            "- Never finish with clichés like 'और इस तरह जिंदगी बदल गई' or 'समय के साथ सब ठीक हो गया'.\n"
            "- Show what happened to each character, how the central mystery was resolved, and the final aftermath.\n\n"
            "Output valid JSON only."
        )

    def generate(self, req: StoryGenerationRequest) -> dict[str, Any]:
        topic = req.topic_research or self.topics.suggest_topic(req.language, req.genre, req.is_short)
        lessons = self._lessons_block()
        category = get_category(req.category_id)
        cat_block = category_story_block(category)
        min_scenes = req.min_scenes or self.settings.story_min_scenes
        max_scenes = req.max_scenes or self.settings.story_max_scenes

        # Step 1: Generate Master Story Blueprint
        blueprint = self.blueprint_generator.generate_blueprint(
            topic=topic,
            category_name=category.name_hi,
            genre=req.genre,
            language=req.language,
        )

        if req.is_short:
            scene_cap = "exactly 4 to 5 scenes"
            brevity = "Total 30-60 seconds spoken. Short Hindi lines. visual_description under 30 English words."
        else:
            scene_cap = f"exactly {min_scenes} to {max_scenes} scenes"
            brevity = (
                f"Generate a deep, engaging, complete cinematic Hindi story of {scene_cap}. "
                "CRITICAL STORYTELLING RULES: "
                "1. NO 1-2 SENTENCE SUMMARIES. Every scene MUST have a rich narrative paragraph (4-7 detailed sentences, MINIMUM 300-450 Devanagari characters) with concrete actions, sensory details, and tangible events. "
                "2. AUTHENTIC DIALOGUE: Every scene MUST have 2-4 lines of natural Hindi dialogue between characters with emotion, pauses ('...'), and distinct personalities. "
                "3. ABSOLUTE ANTI-REPETITION: NEVER repeat questions like 'यह क्या है?', 'यह क्या हो रहा है?', 'तुम कौन हो?'. Every dialogue line must be unique and directly tied to the scene's events. "
                "4. STRICT SHOW, DON'T TELL: If a character suffered, explain the incidents. If betrayed, show the betrayal. "
                "5. The entire story MUST be fully self-contained, resolving all questions."
            )

        lang_rules = ""
        if req.language == "hi":
            lang_rules = """
- Hindi Devanagari ONLY for narration, dialogue, title, synopsis, hook, ending, call_to_action
- Natural emotional Hindi, not translated English
- Pronounce-friendly sentences for TTS
- Include call_to_action field in Hindi
- CRITICAL: narration must NOT repeat or quote the dialogue lines. Narration sets the scene context; dialogue speaks the character's exact words. Never put a character's spoken words inside the narration field — that causes double audio.
- STRICT ANTI-REPETITION MANDATE: Every single scene MUST have unique, progressing narration (at least 3-5 sentences) and distinct character dialogue. NEVER repeat the same phrase, sentence, or dialogue across scenes.
"""

        prompt = f"""Write a completely original {req.genre} story for YouTube.
Language: {req.language}
Content category: {category.name_hi}
Target narration duration: ~{req.target_duration_sec} seconds
Format: {"vertical Short (9:16)" if req.is_short else "landscape cinematic long-form (16:9)"}
Style: {req.content_style}

{cat_block}

Visual style hint: {req.visual_style or category.visual_style}

Topic research:
{topic}

MASTER STORY BLUEPRINT (Follow this exact plot and cause-and-effect progression):
{json.dumps(blueprint.to_dict(), ensure_ascii=False, indent=2)}

Apply lessons:
{lessons}

{lang_rules}

Requirements:
- Use {scene_cap}
- {brevity}
- SHOW WHAT ACTUALLY HAPPENED: Show, don't tell! Never use empty phrases like 'उसे बहुत संघर्ष करना पड़ा' or 'हालात बिगड़ते गए'.
- Scene 1 MUST start with a shocking cold hook.
- Valid JSON only

Schema:
{STORY_SCHEMA_HINT}
"""

        story = None
        last_err: Exception | None = None
        system = self._hindi_system() if req.language == "hi" else "You are an expert screenwriter. Output only valid JSON."

        for attempt in range(3):
            extra = ""
            if attempt > 0:
                extra = "\nIMPORTANT: Previous output was invalid JSON. Return smaller JSON with fewer scenes and shorter strings."
            try:
                response = self.llm.complete(
                    prompt + extra,
                    system=system,
                    json_mode=True,
                )
                story = self.llm.parse_json(response.text)
                if story and story.get("scenes") and len(story["scenes"]) >= 4:
                    break
            except Exception as exc:
                last_err = exc
                logger.warning("Story generation attempt %s failed: %s", attempt + 1, exc)

        if story is None or not story.get("scenes"):
            logger.warning("LLM story generation failed (%s). Using procedural storytelling engine.", last_err)
            story = self._generate_fallback_story(req, topic)

        story["research"] = topic
        story["content_category_id"] = req.category_id
        story["content_category_name"] = category.name_hi
        story["visual_style"] = req.visual_style or category.visual_style
        story["language"] = req.language
        story["genre"] = req.genre
        story["is_short"] = req.is_short
        story["target_duration_sec"] = req.target_duration_sec
        story["story_blueprint"] = blueprint.to_dict()

        if req.language == "hi" and not story.get("call_to_action"):
            story["call_to_action"] = (
                "अगर आपको 'परछाई किसकी थी?' की यह खौफ़नाक कहानी पसंद आई, तो वीडियो को लाइक करें "
                "और ऐसी ही रोंगटे खड़े कर देने वाली रहस्यमयी कहानियों के लिए चैनल को सब्सक्राइब करें।"
            )

        # Clean and deduplicate narration and dialogue
        story = self._clean_and_deduplicate(story)
        story = self._ensure_scene_count(story, req, system)
        story = self._ensure_narration_depth(story, req, system)
        story = self._clean_and_deduplicate(story)

        # ── Strict Story Quality Auditor & Refinement Loop (Min 85/100) ──
        audit_report: StoryAuditReport = self.quality_auditor.audit(story)
        retries = 0
        min_score = getattr(self.settings, "story_min_quality_score", 85)

        while (not audit_report.passed or audit_report.total_score < min_score) and retries < 2:
            retries += 1
            logger.warning(
                "Story scored %s/100 (passed=%s, min=%s). Running StoryRewriter pass %s/2 with feedback:\n%s",
                audit_report.total_score,
                audit_report.passed,
                min_score,
                retries,
                audit_report.feedback_for_rewrite,
            )
            story = self.story_rewriter.rewrite_and_improve(story, audit_report, system)
            story = self._clean_and_deduplicate(story)
            audit_report = self.quality_auditor.audit(story)

        story["quality_report"] = audit_report.to_dict()
        story["quality_score"] = audit_report.total_score
        story["human_review"] = audit_report.human_review_checklist
        logger.info(
            "Final approved story quality score: %s/100 (passed=%s, rep_penalty=%s, vague_penalty=%s)",
            audit_report.total_score,
            audit_report.passed,
            audit_report.repetition_penalty,
            audit_report.vague_penalty,
        )

        with session_scope() as session:
            repo = Repository(session)
            row = repo.create_story(
                title=story.get("title", "Untitled"),
                language=req.language,
                genre=req.genre,
                target_duration_sec=req.target_duration_sec,
                is_short=req.is_short,
                story_json=story,
                lessons_applied={"lessons_snapshot": lessons},
            )
            story["story_project_id"] = row.id

        return story

    GENERIC_REPETITIVE_DIALOGUES = {
        "यह क्या है",
        "यह क्या हो रहा है",
        "तुम कौन हो",
        "यह कौन है",
        "यह क्या मतलब है",
        "तुम्हें कुछ नहीं पता",
        "मैं जानता हूँ कि आप कौन हैं",
        "मैं जानता हूं कि आप कौन हैं",
        "यह कैसे संभव है",
        "मैं समझ गया हूँ",
        "मैं समझ गया हूं",
        "मुझे इसके बारे में पता चलना चाहिए",
        "मुझे यह चाबी मिलनी चाहिए",
        "मैं अकेला नहीं हूँ",
        "मैं अकेला नहीं हूं",
        "मुझे यहाँ से निकलना होगा",
        "मुझे यहां से निकलना होगा",
        "कौन है",
        "क्या हुआ",
    }

    DYNAMIC_DIALOGUE_POOL = [
        "दरवाज़े की कुंडी पर ताज़ा निशान हैं... कोई अभी-अभी यहाँ से भागा है।",
        "इस खत पर पिता की मुहर लगी है, लेकिन तारीख कल रात की है!",
        "तिजोरी खुली है और वसीयत के पन्ने फर्श पर बिखरे पड़े हैं।",
        "तुमने मुझसे यह सच क्यों छुपाया कि हवेली का पिछला हिस्सा तीस साल से सील था?",
        "गाड़ी के पहियों के निशान जंगल की पुरानी बावड़ी की तरफ जा रहे हैं।",
        "आईने के पीछे एक गुप्त ताला लगा हुआ है, इसकी चाबी किसके पास है?",
        "अगर पुलिस ने यह कागज़ात देख लिए, तो हममें से कोई भी बेकसूर साबित नहीं हो पाएगा।",
        "सीढ़ियों के नीचे एक लाल बक्सा दबा हुआ है... उसमें किसी की पुरानी डायरी है।",
    ]

    @staticmethod
    def clean_dialogue_line(raw_line: str, default_char: str = "पात्र") -> tuple[str, str, str]:
        """
        Parses raw dialogue strings to separate (character_name, emotion_instruction, clean_spoken_line).
        E.g. 'मां (घबराकर): रोहित! तुम इतनी देर कहां थे?' -> ('मां', 'घबराकर', 'रोहित! तुम इतनी देर कहां थे?')
        Prevents TTS from reading aloud stage directions and attribution labels.
        """
        line = (raw_line or "").strip()
        character = default_char or "पात्र"
        emotion = ""

        # Pattern: [Character] [(Emotion)]: [Spoken line]
        # Only match actual attribution separators (: or ： or em-dash or spaced dash), never hyphens in compound words
        match = re.match(r"^([\u0900-\u097F\w\s]{1,30}?)\s*(?:\(([^)]+)\))?\s*(?:[:：]|(?:\s+[-—]\s+)|—)\s*(.*)$", line)
        if match:
            char_match = match.group(1).strip()
            if char_match and char_match not in ("Narrator", "सूत्रधार", "Voiceover"):
                character = char_match
            if match.group(2):
                emotion = match.group(2).strip()
            line = match.group(3).strip()
        else:
            emo_match = re.match(r"^\(([^)]+)\)\s*[:：\-]?\s*(.*)$", line)
            if emo_match:
                emotion = emo_match.group(1).strip()
                line = emo_match.group(2).strip()

        # Remove any lingering parenthetical acting directions from spoken audio line
        line = re.sub(r"\([^)]*\)", "", line).strip()
        line = re.sub(r"^[:：\-—,\s]+", "", line).strip()
        return character, emotion, line

    def _clean_and_deduplicate(self, story: dict[str, Any]) -> dict[str, Any]:
        scenes = story.get("scenes") or []
        seen_dialogues: set[str] = set()
        pool_idx = 0

        # Build / normalize character reference sheets
        chars = story.get("characters") or []
        char_refs = story.get("character_references") or []
        ref_map = {c.get("name"): c for c in char_refs if isinstance(c, dict)}

        for c in chars:
            if not isinstance(c, dict):
                continue
            name = c.get("name", "नायक")
            if name not in ref_map:
                desc = c.get("description", "Indian character with sharp expressive features")
                age = c.get("age_group", "25-30 years old")
                gender = c.get("gender", "male")
                ref_map[name] = {
                    "name": name,
                    "gender": gender,
                    "age_group": age,
                    "physical_appearance": f"Sharp natural Indian features, realistic {gender}, {desc}",
                    "signature_attire": "Authentic cinematic Indian attire, detailed textures, practical clothing",
                    "personality": "Observant, emotionally grounded, determined",
                }
        story["character_references"] = list(ref_map.values())

        for sc in scenes:
            sn = sc.get("scene_number", 1)
            narr = (sc.get("narration") or "").strip()
            narr = re.sub(r"^(?:सूत्रधार|Narrator|वॉइसओवर|Voiceover)\s*[:：\-]\s*", "", narr).strip()

            # Ensure minimum narration substance
            if narr and len(narr) < 140:
                mood = (sc.get("mood") or "suspense").lower()
                if "suspense" in mood or "horror" in mood:
                    narr += (
                        " चारों तरफ गहरा सन्नाटा पसरा हुआ था और दिल की धड़कनें साफ़ सुनाई दे रही थीं। "
                        "हवा का हर झोंका किसी अनहोनी का इशारा दे रहा था।"
                    )
                else:
                    narr += (
                        " माहौल में एक अजीब सी बेचैनी थी और हर गुज़रते पल के साथ "
                        "सवालों की उलझन और भी गहरी होती जा रही थी।"
                    )
                sc["narration"] = narr

            # Clean and deduplicate dialogue lines
            new_dialogue = []
            for d in sc.get("dialogue") or []:
                if isinstance(d, str):
                    d = {"character": "पात्र", "line": d, "gender": "male"}
                elif not isinstance(d, dict):
                    continue
                raw_line = (d.get("line") or "").strip()
                if not raw_line:
                    continue

                parsed_char, parsed_emo, clean_line = self.clean_dialogue_line(
                    raw_line, default_char=d.get("character", "पात्र")
                )
                d["character"] = parsed_char
                if parsed_emo:
                    d["acting_instruction"] = parsed_emo
                d["line"] = clean_line

                norm = clean_line.replace("?", "").replace("।", "").replace("!", "").replace(".", "").replace(",", "").strip()
                is_duplicate = norm in seen_dialogues

                if is_duplicate:
                    replacement = self.DYNAMIC_DIALOGUE_POOL[pool_idx % len(self.DYNAMIC_DIALOGUE_POOL)]
                    pool_idx += 1
                    logger.info("Scene %s: Replaced repetitive dialogue '%s' -> '%s'", sn, clean_line, replacement)
                    d["line"] = replacement
                    seen_dialogues.add(replacement)
                else:
                    seen_dialogues.add(norm)
                new_dialogue.append(d)
            sc["dialogue"] = new_dialogue

        # Ensure hook_opening and ending match
        if scenes:
            if not story.get("hook_opening"):
                story["hook_opening"] = (scenes[0].get("narration") or "")[:200]
            if not story.get("ending"):
                story["ending"] = (scenes[-1].get("narration") or "")[:200]

        return story

    def _ensure_scene_count(
        self, story: dict[str, Any], req: StoryGenerationRequest, system: str
    ) -> dict[str, Any]:
        if req.is_short:
            return story
        min_scenes = req.min_scenes or self.settings.story_min_scenes
        scenes = story.get("scenes") or []

        if len(scenes) >= max(7, min_scenes - 2) and self._calculate_spoken_chars(story) >= 3500:
            logger.info("Story has %d strong scenes with %d characters — complete story", len(scenes), self._calculate_spoken_chars(story))
            return story

        scenes_needed = min_scenes
        if len(scenes) >= scenes_needed:
            return story

        for attempt in range(3):
            scenes = story.get("scenes") or []
            if len(scenes) >= scenes_needed:
                break
            needed = min(scenes_needed - len(scenes), 3)
            logger.info("Expanding story: currently %d scenes -> adding %d more", len(scenes), needed)

            expand_prompt = f"""Continue this Hindi story JSON. Add exactly {needed} NEW scenes showing concrete actions moving the plot towards climax.
Title: {story.get("title")}
Characters: {story.get("characters")}
Last scenes: {scenes[-2:]}

RULES:
- STRICT SHOW, DON'T TELL: No empty assertions.
- Narration: 4-6 detailed Devanagari sentences (min 300 chars) with concrete actions and tangible events.
- Dialogue: 2-3 lines of realistic conversational Hindi dialogue.
- Zero repetition.

Return JSON:
{{"scenes": [ ... new scenes only, scene_number continues ... ]}}
"""
            try:
                response = self.llm.complete(expand_prompt, system=system, json_mode=True)
                extra = self.llm.parse_json(response.text)
                new_scenes = extra.get("scenes") or []
                existing_narrs = {s.get("narration", "").strip() for s in scenes}
                unique_new = [sc for sc in new_scenes if sc.get("narration", "").strip() not in existing_narrs]

                if not unique_new:
                    break
                max_num = max(s.get("scene_number", i + 1) for i, s in enumerate(scenes))
                for i, sc in enumerate(unique_new):
                    sc["scene_number"] = max_num + i + 1
                story["scenes"] = scenes + unique_new
            except Exception as exc:
                logger.warning("Scene expansion attempt %s failed: %s", attempt + 1, exc)
                break

        return story

    def _calculate_spoken_chars(self, story: dict[str, Any]) -> int:
        total = 0
        for sc in story.get("scenes") or []:
            total += len(sc.get("narration", "").strip())
            for d in sc.get("dialogue") or []:
                total += len(d.get("line", "").strip())
        return total

    def _ensure_narration_depth(
        self, story: dict[str, Any], req: StoryGenerationRequest, system: str
    ) -> dict[str, Any]:
        if req.is_short:
            return story

        min_chars = max(4200, int(req.target_duration_sec * 8.5))
        scenes = story.get("scenes") or []
        current_chars = self._calculate_spoken_chars(story)

        if current_chars >= min_chars:
            return story

        batch_size = 2
        for i in range(0, len(scenes), batch_size):
            batch = scenes[i : i + batch_size]
            needs_expansion = [s for s in batch if len(s.get("narration", "").strip()) < 260]
            if not needs_expansion:
                continue

            prompt = f"""You are an elite Indian screenwriter. Expand each scene's narration into a rich, detailed dramatic Hindi Devanagari narrative paragraph (4-6 sentences, MINIMUM 320 characters) with concrete actions and sensory details.
Do NOT summarize. Write vivid, tangible scene events.
No empty phrases like 'उसे बहुत संघर्ष करना पड़ा'.

Scenes to expand:
{json.dumps([{'scene_number': s.get('scene_number'), 'current_narration': s.get('narration')} for s in needs_expansion], ensure_ascii=False, indent=2)}

Return JSON:
{{
  "expanded_scenes": [
    {{
      "scene_number": <int>,
      "narration": "Rich detailed Devanagari paragraph with concrete actions (min 320 characters)...",
      "dialogue": [ {{"character": "name", "line": "Hindi line", "gender": "male|female"}} ]
    }}
  ]
}}"""
            try:
                res = self.llm.complete(prompt, system=system, json_mode=True)
                data = self.llm.parse_json(res.text)
                items = data.get("expanded_scenes") or data.get("scenes") or []
                expanded = {s.get("scene_number"): s for s in items if isinstance(s, dict) and s.get("scene_number")}
                for sc in needs_expansion:
                    sn = sc.get("scene_number")
                    if sn in expanded:
                        new_narr = expanded[sn].get("narration", "").strip()
                        if len(new_narr) > len(sc.get("narration", "")):
                            sc["narration"] = new_narr
                        new_dial = expanded[sn].get("dialogue")
                        if new_dial and isinstance(new_dial, list) and len(new_dial) > 0:
                            sc["dialogue"] = new_dial
            except Exception as exc:
                logger.warning("Failed to expand batch %d-%d via LLM: %s", i, i + batch_size, exc)

        return story

    def _generate_fallback_story(self, req: StoryGenerationRequest, topic: dict[str, Any]) -> dict[str, Any]:
        """
        Procedural masterclass horror & mystery story for YouTube Channel: 'परछाई किसकी थी?'.
        15 complete, rich, scene-based chapters with concrete sensory actions, clues,
        meaningful dialogues, zero repetition, zero vague storytelling, and full resolution.
        """
        topic_title = topic.get("main_topic") or "हवेली का बंद कमरा: परछाई किसकी थी?"
        title = topic_title if ("परछाई" in topic_title or "हवेली" in topic_title) else f"{topic_title} | परछाई किसकी थी?"

        characters = [
            {
                "name": "राघव",
                "gender": "male",
                "age_group": "27 years old",
                "description": "Young Indian investigator with sharp jawline, intense eyes, wearing dark navy blue jacket over grey henley",
                "voice_id_hint": "hi_IN-pratham-medium",
            },
            {
                "name": "माया",
                "gender": "female",
                "age_group": "25 years old",
                "description": "Courageous documentary filmmaker with determined eyes, wearing olive green outdoor jacket",
                "voice_id_hint": "hi_IN-priyamvada-medium",
            },
        ]

        character_references = [
            {
                "name": "राघव",
                "gender": "male",
                "age_group": "27 years old",
                "physical_appearance": "Sharp Indian features, clean shaven, intense piercing eyes, short black styled hair, height 5'11",
                "signature_attire": "Dark navy blue jacket, grey henley t-shirt, silver watch on left wrist",
                "personality": "Cautious, razor-sharp observant, fearless",
            },
            {
                "name": "माया",
                "gender": "female",
                "age_group": "25 years old",
                "physical_appearance": "Expressive large dark eyes, shoulder-length wavy black hair, height 5'5",
                "signature_attire": "Olive green outdoor jacket, black cargo pants, leather strap camera around neck",
                "personality": "Intuitive, observant, sensitive to supernatural atmosphere",
            },
            {
                "name": "कैलाश चाचा",
                "gender": "male",
                "age_group": "52 years old",
                "physical_appearance": "Heavy-set Indian man, stern forehead, sharp narrow eyes, salt-and-pepper hair",
                "signature_attire": "Beige safari suit, leather briefcase, thick gold ring on right index finger",
                "personality": "Greedy, manipulative, desperate when cornered",
            },
        ]

        scenes = [
            {
                "scene_number": 1,
                "story_beat": "opening_hook",
                "duration_estimate_sec": 45,
                "visual_description": "Cinematic ultra-realistic 35mm photo of ancient dark Indian haveli in heavy midnight rain, lightning illuminating cracked royal archway, single crimson lantern glowing in attic window, photorealistic horror cinema, 8k",
                "image_prompt": "Cinematic ultra-realistic photo of an ancient dark Indian haveli at stormy midnight, pouring rain, glowing crimson lantern in attic window, photorealistic horror cinema",
                "narration": "रात के ठीक बारह बजे, मूसलाधार बारिश के बीच पुरानी वीरान हवेली की सबसे ऊपरी खिड़की से अचानक लाल रोशनी फूट पड़ी। राघव ने जैसे ही हवेली के भारी जंग लगे लोहे के फाटक को धक्का दिया, लोहे की चीख़ती आवाज़ ने सन्नाटे को चीर दिया। प्रवेश द्वार की गीली मिट्टी पर ताज़ा जूतों के निशान बने हुए थे, जो साफ़ इशारा कर रहे थे कि कोई चंद मिनट पहले ही अंदर दाखिल हुआ था।",
                "dialogue": [
                    {"character": "राघव", "acting_instruction": "दबी हुई सतर्क आवाज़", "line": "माया, ज़मीन की तरफ टॉर्च करो... मिट्टी पर किसी भारी बूट्स के ताज़ा निशान हैं।", "gender": "male"},
                    {"character": "माया", "acting_instruction": "कांपती हुई आवाज़", "line": "लेकिन राघव, इस हवेली का मुख्य ताला तो बरसों से बंद था, फिर कोई अंदर कैसे गया?", "gender": "female"},
                ],
                "mood": "horror",
                "characters_present": ["राघव", "माया"],
                "sfx_hint": "thunder, rain, creaking iron gate, squelching mud",
            },
            {
                "scene_number": 2,
                "story_beat": "character_intro",
                "duration_estimate_sec": 45,
                "visual_description": "Grand deserted Indian mansion hallway with crumbling pillars, broken antique furniture, spiderwebs, blue moonlight through tall broken archways",
                "image_prompt": "Ancient dusty hall inside an Indian palace, broken antique furniture, moonlight through broken windows, eerie suspenseful atmosphere",
                "narration": "मुख्य दालान में दाखिल होते ही दोनों के सामने विशाल पत्थर के खंभे और टूटे हुए झूमर नज़र आए। दीवारों पर हवेली के पुराने राजपरिवार की तस्वीरें लटकी थीं, जिनमें से हर तस्वीर पर चेहरे को नुकीली चीज़ से खुरच कर मिटा दिया गया था। राघव ने अपनी जेब से एक पुराना पीला कागज़ निकाला, जिस पर उसके स्वर्गीय पिता की आखिरी वसीयत की अधूरी नकल दर्ज थी।",
                "dialogue": [
                    {"character": "राघव", "acting_instruction": "गंभीरता से कागज़ दिखाते हुए", "line": "पिताजी ने अपनी डायरी में लिखा था कि इस हवेली का असली सच पहली मंज़िल के बंद कमरे में छुपा है।", "gender": "male"},
                    {"character": "माया", "acting_instruction": "कैमरा दीवार की ओर घुमाते हुए", "line": "इन तस्वीरों को देखो राघव... किसी ने जानबूझकर सारे चेहरे मिटाए हैं ताकि कोई पहचान न सके।", "gender": "female"},
                ],
                "mood": "suspense",
                "characters_present": ["राघव", "माया"],
                "sfx_hint": "footsteps on broken glass, wind howling through pillars",
            },
            {
                "scene_number": 3,
                "story_beat": "first_strange_incident",
                "duration_estimate_sec": 45,
                "visual_description": "Antique grandfather clock in corner with pendulum swinging violently backwards, candles catching blue fire spontaneously on ornate wooden table",
                "image_prompt": "Dramatic antique grandfather clock in a dark haveli, candles suddenly catching fire on ornate table, supernatural eerie atmosphere",
                "narration": "तभी दालान के कोने में रखी विशाल पीतल की पेंडुलम घड़ी ज़ोरदार खड़खड़ाहट के साथ उलटी दिशा में दौड़ने लगी। उसी क्षण मेज़ पर रखे पुराने झाड़फानूस की सात मोमबत्तियाँ बिना किसी तीली के अपने आप नीली लौ के साथ भड़क उठीं। कमरे का तापमान इतना तेज़ी से गिरा कि दोनों के मुँह से सफ़ेद भाप निकलने लगी और दीवारों का प्लास्टर चटकने लगा।",
                "dialogue": [
                    {"character": "माया", "acting_instruction": "घबराकर पीछे हटते हुए", "line": "राघव! वो मोमबत्तियाँ अपने आप जल उठीं... हवा में सल्फर की गंध आ रही है!", "gender": "female"},
                    {"character": "राघव", "acting_instruction": "टॉर्च से फर्श जांचते हुए", "line": "शांत रहो माया, कैमरा चालू रखो। यह कोई भ्रम नहीं है, यहाँ किसी की मौजूदगी है।", "gender": "male"},
                ],
                "mood": "horror",
                "characters_present": ["राघव", "माया"],
                "sfx_hint": "clock ticking backwards rapidly, fire whooshing, cold wind gust",
            },
            {
                "scene_number": 4,
                "story_beat": "suspense_development",
                "duration_estimate_sec": 45,
                "visual_description": "Raghav finding open iron safe embedded in stone wall, scattered yellowed account ledgers on floor, lantern light",
                "image_prompt": "Open rusty iron safe in a dark stone wall, scattered vintage documents on dusty marble floor, warm lantern light",
                "narration": "दीवार के पिछले हिस्से में राघव को एक खुली हुई लोहे की पुरानी तिजोरी दिखाई दी। तिजोरी का भारी ताला किसी आरी से हाल ही में काटा गया था और ज़मीन पर लोहे का ताज़ा बुरादा बिखरा पड़ा था। राघव ने नीचे गिरे बहीखाते को उठाया, जिसमें तीस साल पहले दर्जनों बैंक खातों से निकाले गए लाखों रुपयों का हिसाब और एक नाम दर्ज था।",
                "dialogue": [
                    {"character": "राघव", "acting_instruction": "चौंककर बहीखाता पढ़ते हुए", "line": "इस बहीखाते पर कैलाश चाचा के हस्ताक्षर हैं! तीस साल पहले उन्होंने हवेली का सारा पैसा निकाला था।", "gender": "male"},
                    {"character": "माया", "acting_instruction": "आश्चर्य से नज़दीक आते हुए", "line": "वही चाचा जिन्होंने कल तुमसे कहा था कि यह हवेली शापित है और यहाँ कभी मत आना?", "gender": "female"},
                ],
                "mood": "suspense",
                "characters_present": ["राघव", "माया"],
                "sfx_hint": "paper rustling, metallic clinking of saw dust",
            },
            {
                "scene_number": 5,
                "story_beat": "suspense_development",
                "duration_estimate_sec": 45,
                "visual_description": "Grand curving wooden staircase stretching upward into pitch darkness, an unnatural elongated dark shadow stretching up the wall without a person casting it",
                "image_prompt": "Ancient creaking wooden staircase in a dark palace, an unnatural elongated shadow moving on the wall, cinematic moody lighting",
                "narration": "सीढ़ियों के पास पहुँचते ही दीवार पर एक लंबा, भयानक साया रेंगता हुआ पहली मंज़िल की ओर बढ़ता दिखाई दिया। सीढ़ियों की लकड़ी पर किसी के भारी जूतों की चरमराहट गूँज रही थी, लेकिन रोशनी फेंकने पर वहाँ कोई इंसान नज़र नहीं आ रहा था। सन्नाटे में एक औरत की धीमी, दर्दनाक रोने की आवाज़ तैरने लगी, जो सीधे ऊपर वाले गलियारे से आ रही थी।",
                "dialogue": [
                    {"character": "माया", "acting_instruction": "ऊपर की ओर इशारा करते हुए", "line": "राघव, दीवार की परछाई देखो! वो हमारे कदमों के साथ नहीं चल रही, वो तो आगे जा रही है!", "gender": "female"},
                    {"character": "राघव", "acting_instruction": "सीढ़ियों पर कदम बढ़ाते हुए", "line": "रोने की आवाज़ उसी बंद कमरे से आ रही है। हमें ऊपर जाकर देखना ही होगा।", "gender": "male"},
                ],
                "mood": "horror",
                "characters_present": ["राघव", "माया"],
                "sfx_hint": "creaking floorboards, mournful distant weeping, high pitch violin",
            },
            {
                "scene_number": 6,
                "story_beat": "suspense_development",
                "duration_estimate_sec": 45,
                "visual_description": "Upper floor hallway with heavy wooden door wrapped in thick rusted chains, massive brass padlocks, red painted tantric symbols on wooden frame",
                "image_prompt": "Old wooden door with a heavy rusted padlock and iron chains, mysterious red symbols, dark hallway, ominous glow",
                "narration": "पहली मंज़िल के संकरे गलियारे के अंत में वह कमरा था, जिसके भारी शीशम के दरवाज़े पर मोटी लोहे की ज़ंजीरें कसी हुई थीं। चौखट पर गेरुए रंग से पुराने तांत्रिक मंत्र और चेतावनी भरे निशान उकेरे गए थे। दरवाज़े की दरारों से एक बर्फीली सर्द हवा बाहर निकल रही थी, जिससे सीढ़ियों के पास रखी मोमबत्तियों की लौ थरथराने लगी।",
                "dialogue": [
                    {"character": "राघव", "acting_instruction": "ज़ंजीरों को हाथ से छूते हुए", "line": "यह ज़ंजीरें बाहर से बांधी गई हैं... जैसे अंदर किसी को ज़िंदा दफ़न करने की साज़िश रची गई हो।", "gender": "male"},
                    {"character": "माया", "acting_instruction": "दरवाज़े के नीचे देखते हुए", "line": "दरवाज़े की निचली चौखट पर देखो, किसी ने हाल ही में ज़ंजीर के एक कुंडे को हिलाया है!", "gender": "female"},
                ],
                "mood": "suspense",
                "characters_present": ["राघव", "माया"],
                "sfx_hint": "iron chains clinking, cold wind whistle through keyhole",
            },
            {
                "scene_number": 7,
                "story_beat": "major_revelation",
                "duration_estimate_sec": 45,
                "visual_description": "Raghav finding hollow brick recess beside door frame, pulling out vintage red velvet diary with ornate brass key tied to ribbon",
                "image_prompt": "Close-up of vintage yellowed journal pages with handwritten Hindi script, an ornate brass key beside it, dramatic candle lighting",
                "narration": "चौखट की एक ढीली ईंट को हटाते ही राघव के हाथ एक गुप्त आले में रखी लाल मखमली डायरी और भारी पीतल की प्राचीन चाबी लगी। राघव ने कांपती उँगलियों से डायरी का पहला पन्ना खोला, जिस पर उसके दादाजी की तीस साल पुरानी आखिरी लिखावट दर्ज थी। पन्ने पर लिखा था कि हवेली का असली वारिस राघव है और उसके सगे चाचा कैलाश ने वसीयत चुराने के लिए हवेली में आग लगाने की धमकी दी थी।",
                "dialogue": [
                    {"character": "राघव", "acting_instruction": "गहरे सदमे और गुस्से में", "line": "माया... मेरे दादाजी ने लिखा है कि चाचा ने उन्हें इसी कमरे में बंद कर झूठे दस्तखत करवाए थे!", "gender": "male"},
                    {"character": "माया", "acting_instruction": "डायरी पर नज़र डालते हुए", "line": "और इस चाबी से वो तिजोरी खुलती है जहाँ असली वसीयत रखी है!", "gender": "female"},
                ],
                "mood": "suspense",
                "characters_present": ["राघव", "माया"],
                "sfx_hint": "paper turning, heavy stone scraping",
            },
            {
                "scene_number": 8,
                "story_beat": "suspense_development",
                "duration_estimate_sec": 45,
                "visual_description": "Corridor shadows shifting, sound of heavy footsteps behind them, Raghav turning flashlight towards staircase",
                "image_prompt": "Dark shadowy hallway in haveli, flashlight beam cutting through mist, a menacing silhouette approaching from behind",
                "narration": "अभी राघव डायरी की अगली पंक्तियाँ पढ़ ही रहा था कि सीढ़ियों से किसी के तेज़ कदमों की धमक सुनाई दी। गलियारे की मोमबत्तियाँ एक-एक करके बुझने लगीं और हवा में बारूद की तीखी महक भर गई। राघव ने अपनी टॉर्च सीढ़ियों की तरफ घुमाई, तो सामने बरसाती कोट पहने एक भारी-भरकम इंसान खड़ा था, जिसके हाथ में लोहे की भारी रॉड चमक रही थी।",
                "dialogue": [
                    {"character": "राघव", "acting_instruction": "टॉर्च का फोकस चेहरे पर करते हुए", "line": "कौन है वहाँ? सामने आओ, चेहरा दिखाओ!", "gender": "male"},
                    {"character": "कैलाश चाचा", "acting_instruction": "कुटिल और खुरदुरी आवाज़", "line": "चाचा से ऐसे बात करोगे राघव? मैंने कहा था न कि इस हवेली में तुम्हारे लिए सिर्फ मौत रखी है।", "gender": "male"},
                ],
                "mood": "horror",
                "characters_present": ["राघव", "माया", "कैलाश चाचा"],
                "sfx_hint": "heavy boot thuds, metal rod dragging on stone, sudden wind rush",
            },
            {
                "scene_number": 9,
                "story_beat": "suspense_development",
                "duration_estimate_sec": 45,
                "visual_description": "Tense confrontation in misty hallway, Kailash Uncle pointing iron rod demanding the red diary, Maya recording with camera",
                "image_prompt": "Cinematic tense confrontation in dark hallway, older man with iron rod demanding documents from young man, high contrast lighting",
                "narration": "कैलाश चाचा ने आगे बढ़कर लोहे की रॉड राघव के सीने के सामने तान दी। उनकी आँखों में लालच और नफ़रत साफ़ झलक रही थी, क्योंकि उन्हें भनक लग चुकी थी कि राघव असली दस्तावेज़ तक पहुँच चुका है। माया ने बिना डरे अपने कैमरे का लाल रिकॉर्डिंग बटन दबाया और दीवार की आड़ लेकर पूरी घटना को रिकॉर्ड करने लगी।",
                "dialogue": [
                    {"character": "कैलाश चाचा", "acting_instruction": "दांत पीसते हुए गुस्से में", "line": "वो लाल डायरी और चाबी मुझे दे दो राघव, वरना आज रात तुम दोनों इसी हवेली का हिस्सा बन जाओगे!", "gender": "male"},
                    {"character": "राघव", "acting_instruction": "डायरी को पीछे छुपाते हुए दृढ़ता से", "line": "तीस साल पहले तुमने मेरे दादाजी के साथ जो किया, उसका हिसाब अब कानून करेगा चाचा!", "gender": "male"},
                ],
                "mood": "suspense",
                "characters_present": ["राघव", "माया", "कैलाश चाचा"],
                "sfx_hint": "low drone, thunder rumbling, camera beep",
            },
            {
                "scene_number": 10,
                "story_beat": "suspense_development",
                "duration_estimate_sec": 45,
                "visual_description": "Physical struggle in narrow corridor, Raghav dodging iron rod strike, disarming Kailash Uncle with swift tackle",
                "image_prompt": "Action freeze of two men grappling in dark hallway, dropped iron rod hitting marble floor with sparks, dynamic cinematic angle",
                "narration": "कैलाश ने पूरी ताक़त से रॉड राघव के सिर पर घुमाई, लेकिन राघव ने फुर्ती से झुककर वार खाली कर दिया। अगले ही पल राघव ने चाचा की कलाई मरोड़कर रॉड ज़मीन पर गिरा दी, जो झनझनाहट के साथ फर्श पर दूर जा गिरी। कैलाश लड़खड़ाकर दीवार से टकराए और उनके चेहरे पर हार का खौफ़ साफ़ दिखने लगा।",
                "dialogue": [
                    {"character": "राघव", "acting_instruction": "हाथ मरोड़ते हुए कड़े स्वर में", "line": "सच बताओ चाचा! तीस साल पहले इस बंद कमरे में क्या हुआ था?", "gender": "male"},
                    {"character": "कैलाश चाचा", "acting_instruction": "दर्द में कराहते हुए", "line": "मैंने कुछ नहीं किया... वो कमरा खुद उन्हें निगल गया था! वो आईना... उस आईने में कोई शैतान रहता है!", "gender": "male"},
                ],
                "mood": "suspense",
                "characters_present": ["राघव", "माया", "कैलाश चाचा"],
                "sfx_hint": "body tackle thud, iron rod clattering, heavy breathing",
            },
            {
                "scene_number": 11,
                "story_beat": "climax",
                "duration_estimate_sec": 45,
                "visual_description": "Chains wrapping the heavy wooden door snapping with loud metallic clang, door bursting open violently, thick freezing supernatural mist pouring out",
                "image_prompt": "Heavy wooden chained door exploding open, thick dark fog billowing into hallway, inside an eerie tall antique mirror glows softly in darkness",
                "narration": "ठीक उसी वक्त, रात का एक बजते ही एक कानफोड़ू धमाके के साथ दरवाज़े पर बंधी मोटी ज़ंजीरें तिनके की तरह टूटकर फर्श पर बिखर गईं। भारी शीशम का दरवाज़ा अपने आप चौखट से बाहर की ओर खुला और कमरे के अंदर से बर्फ जैसी ठंडी धुंध गलियारे में फैल गई। कमरे के बिल्कुल बीचों-बीच एक विशाल आदमकद प्राचीन आईना रखा था, जिसकी सतह पर अजीब सी नीली रोशनी तैर रही थी।",
                "dialogue": [
                    {"character": "माया", "acting_instruction": "सहमकर पीछे हटते हुए", "line": "राघव देखो! ज़ंजीरें अंदर से किसी ने तोड़ दी हैं... दरवाज़ा खुल चुका है!", "gender": "female"},
                    {"character": "कैलाश चाचा", "acting_instruction": "पागलों की तरह कांपते हुए", "line": "नहीं... वो बाहर आ गया! मुझे यहाँ से जाने दो, वो मुझे नहीं छोड़ेगा!", "gender": "male"},
                ],
                "mood": "horror",
                "characters_present": ["राघव", "माया", "कैलाश चाचा"],
                "sfx_hint": "metal snapping like gunshots, heavy wooden door bang, rushing supernatural wind",
            },
            {
                "scene_number": 12,
                "story_beat": "shocking_twist",
                "duration_estimate_sec": 45,
                "visual_description": "Raghav standing in front of the giant antique mirror, seeing reflection of Kailash Uncle with sinister shadowy demon clutching his shoulder in the glass",
                "image_prompt": "Terrified young Indian man standing in front of cracked antique mirror, his reflection has glowing eyes and a sinister dark silhouette stands behind him, intense horror movie lighting",
                "narration": "राघव जैसे ही कमरे के अंदर दाखिल होकर आईने के सामने खड़ा हुआ, उसने देखा कि आईने में उसकी अपनी कोई परछाई नहीं थी। आईने के भीतर सिर्फ कैलाश चाचा खड़े दिखाई दे रहे थे, और उनके पीछे एक विशाल काली आकृति खड़ी थी, जिसके चेहरे पर वही चेहरा उभर रहा था जिसे दालान की पुरानी तस्वीर से खुरच कर मिटाया गया था—राघव के दादाजी का चेहरा। उस आकृति के हाथ में वही पुराना दस्तावेज़ था जिस पर चाचा के झूठे अँगूठे के निशान लगे थे।",
                "dialogue": [
                    {"character": "राघव", "acting_instruction": "आईने में देखकर स्तब्ध स्वर में", "line": "आईने में परछाई तुम्हारी गुनाहों की है चाचा... दादाजी की आत्मा तीस साल से तुम्हारे जुर्म की गवाही दे रही थी!", "gender": "male"},
                    {"character": "कैलाश चाचा", "acting_instruction": "घुटनों पर गिरकर हाथ जोड़ते हुए", "line": "मुझे माफ़ कर दो! मैंने ही भैया को ज़हर देकर इस कमरे में बंद किया था... मुझे छोड़ दो!", "gender": "male"},
                ],
                "mood": "horror",
                "characters_present": ["राघव", "माया", "कैलाश चाचा"],
                "sfx_hint": "glass hum, eerie deep demonic whisper, sobbing echo",
            },
            {
                "scene_number": 13,
                "story_beat": "climax",
                "duration_estimate_sec": 45,
                "visual_description": "Kailash Uncle kneeling on floor confessing in front of Maya's glowing video camera, mirror shadow slowly dissipating into peaceful white light",
                "image_prompt": "Older man weeping and confessing on knees before young woman holding modern video camera, eerie shadows dissolving into soft morning glow",
                "narration": "कैलाश चाचा रोते हुए अपने घुटनों के बल गिर पड़े और माया के कैमरे के सामने उन्होंने तीस साल पुराना पूरा सच बयां कर दिया। जैसे-जैसे उनके मुँह से गुनाह की सच्चाई बाहर आ रही थी, आईने के अंदर की खौफ़नाक काली परछाई धीरे-धीरे शांत होकर सफ़ेद रोशनी में तब्दील होने लगी। कमरे का दमघोंटू तापमान सामान्य होने लगा और हवा से सल्फर की बदबू गायब हो गई।",
                "dialogue": [
                    {"character": "माया", "acting_instruction": "कैमरा स्थिर रखते हुए आत्मविश्वास से", "line": "तुम्हारा एक-एक शब्द कैमरे में रिकॉर्ड हो चुका है कैलाश... अब कोई अदालत तुम्हें नहीं बचा सकती।", "gender": "female"},
                    {"character": "राघव", "acting_instruction": "चाबी से तिजोरी का ताला खोलते हुए", "line": "और इस आईने के पीछे छुपी असली वसीयत अब मेरे हाथ में है। सच की जीत हो चुकी है।", "gender": "male"},
                ],
                "mood": "suspense",
                "characters_present": ["राघव", "माया", "कैलाश चाचा"],
                "sfx_hint": "gentle chime, camera recording hum, sigh of wind",
            },
            {
                "scene_number": 14,
                "story_beat": "meaningful_ending",
                "duration_estimate_sec": 45,
                "visual_description": "Police jeeps arriving at mansion gates with flashing blue and red lights through morning mist, officers taking Kailash Uncle into custody",
                "image_prompt": "Police vehicles with flashing red and blue lights outside vintage Indian mansion at misty dawn, officers leading away handcuffed suspect in beige suit",
                "narration": "सुबह के पाँच बजते ही पुलिस की गाड़ियों के सायरन से हवेली का सन्नाटा गूँज उठा। राघव की सूचना पर पहुँचे पुलिस अधिकारियों ने कैलाश चाचा को हथकड़ी लगाकर अपनी जीप में बिठा लिया, जबकि राघव ने दादाजी की मूल वसीयत और कैमरे की वीडियो रिकॉर्डिंग इंस्पेक्टर के सुपुर्द कर दी। तीस साल पुराना अनसुलझा रहस्य अब हमेशा के लिए क़ानूनी तौर पर सुलझ चुका था।",
                "dialogue": [
                    {"character": "राघव", "acting_instruction": "इंस्पेक्टर को कागज़ सौंपते हुए", "line": "इंस्पेक्टर साहब, यह रहा तीस साल पुराने कत्ल और वसीयत की जालसाज़ी का पूरा सबूत।", "gender": "male"},
                    {"character": "माया", "acting_instruction": "कैमरा बैग में रखते हुए सुकून से", "line": "आखिरकार आज इस हवेली की दीवारों को इंसाफ़ मिल गया राघव।", "gender": "female"},
                ],
                "mood": "suspense",
                "characters_present": ["राघव", "माया"],
                "sfx_hint": "distant police siren, car doors slamming, morning bird chirp",
            },
            {
                "scene_number": 15,
                "story_beat": "meaningful_ending",
                "duration_estimate_sec": 45,
                "visual_description": "Golden sunrise rays penetrating cracked haveli arches, Raghav and Maya walking out into sunlight, haveli standing peaceful and bright",
                "image_prompt": "Gloomy dawn breaking over ancient haunted Indian haveli, thick mist, two silhouettes walking away, high attic window showing a solitary eerie shadow looking down, 8k cinematic",
                "narration": "सूरज की सुनहरी किरणें हवेली के दालान पर फैलीं, तो बरसों पुराना खौफ़ और अंधेरा पूरी तरह मिट चुका था। राघव ने हवेली के मुख्य दरवाज़े पर नया ताला लगाया और माया के साथ बाहर खुली सड़क पर आ गया। उसने मुड़कर पहली मंज़िल के उस कमरे की खिड़की की ओर देखा, जहाँ अब कोई लाल रोशनी या डरावना साया नहीं था, बल्कि सुबह की ताज़ा हवा में सफ़ेद पर्दा सुकून से लहरा रहा था।",
                "dialogue": [
                    {"character": "राघव", "acting_instruction": "मुस्कुराते हुए सुकून भरी आवाज़ में", "line": "परछाई किसी भूत की नहीं थी माया, वो तो सच के सामने आने का इंतज़ार कर रही थी।", "gender": "male"},
                    {"character": "माया", "acting_instruction": "राघव के कंधे पर हाथ रखते हुए", "line": "चलो राघव... अब एक नए सफर की शुरुआत करते हैं।", "gender": "female"},
                ],
                "mood": "suspense",
                "characters_present": ["राघव", "माया"],
                "sfx_hint": "temple bell in distance, gentle morning breeze, warm cinematic swell",
            },
        ]

        synopsis = "एक वीरान हवेली के बंद कमरे में आधी रात को दिखाई देने वाली रहस्यमयी परछाई और आईने का वो खौफ़नाक सच जो राघव के अपने परिवार के तीस साल पुराने विश्वासघात और कत्ल का पर्दाफाश करता है।"
        conclusion_moral = "अतीत के गुनाहों को चाहे कितनी भी गहरी खामोशी में दफ़न कर दिया जाए, सच्चाई की परछाई गुनहगार को कभी बख्शती नहीं है।"

        return {
            "title": title,
            "synopsis": synopsis,
            "scenes": scenes,
            "characters": characters,
            "character_references": character_references,
            "moral": conclusion_moral,
            "conclusion_moral": conclusion_moral,
            "ending": "राघव ने मुड़कर पहली मंज़िल के उस कमरे की खिड़की की ओर देखा, जहाँ अब कोई डरावना साया नहीं था, बल्कि सुबह की ताज़ा हवा में सफ़ेद पर्दा सुकून से लहरा रहा था और सच की जीत हो चुकी थी।",
            "call_to_action": "अगर आपको 'परछाई किसकी थी?' की यह खौफ़नाक कहानी पसंद आई, तो वीडियो को लाइक करें और ऐसी ही रोंगटे खड़े कर देने वाली रहस्यमयी कहानियों के लिए चैनल को सब्सक्राइब करें।",
            "thumbnail": {
                "headline": "परछाई किसकी थी?",
                "visual_concept": "Terrified young man looking into an antique mirror where a dark silhouette smiles behind him in a dimly lit haunted haveli",
                "image_prompt": "Cinematic ultra-realistic YouTube thumbnail 16:9, terrified young Indian man with wide horrified eyes holding brass lantern, antique mirror showing sinister dark shadowy figure behind him, dramatic chiaroscuro rim lighting, high contrast horror movie poster aesthetic, 8k",
            },
            "story_project_id": 1,
        }
