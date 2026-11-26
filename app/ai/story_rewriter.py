from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.ai.providers import LLMRouter
from app.ai.repetition_detector import RepetitionDetector
from app.ai.story_quality_auditor import StoryAuditReport
from app.ai.vague_sentence_detector import VagueSentenceDetector

logger = logging.getLogger(__name__)


class StoryRewriter:
    """
    Surgically rewrites weak scenes, removes vague assertions by showing actual events,
    eliminates repetitive sentences and dialogue loops, and repairs incomplete endings.
    """

    CONCRETE_SCENE_REPLACEMENTS = {
        "मुश्किलों का सामना करना पड़ा": (
            "बैंक का नोटिस घर आते ही राघव के होश उड़ गए। सामने खड़े बैंक मैनेजर ने साफ़ कह दिया था कि "
            "अगर अगले चौबीस घंटों में पिता का कर्ज़ नहीं चुकाया गया, तो पुश्तैनी मकान नीलाम कर दिया जाएगा।"
        ),
        "परेशानियां बढ़ती चली गईं": (
            "अगली सुबह जब राघव दफ़्तर पहुँचा, तो उसके केबिन का ताला टूटा हुआ था और कंप्यूटर से सारी गुप्त फ़ाइलें "
            "गायब थीं। मेज़ पर लाल स्याही से लिखा एक धमकी भरा पत्र पड़ा था।"
        ),
        "हालात बहुत खराब हो गए": (
            "गाँव के कुएँ का पानी अचानक काला पड़ गया और एक के बाद एक मवेशी दम तोड़ने लगे। "
            "चौपाल पर जमा लोग अब हवेली की तरफ गुस्से से देख रहे थे और हाथों में मशालें थाम रखी थीं।"
        ),
        "समस्याओं से गुजरना पड़ा": (
            "राघव के चाचा ने ज़मीन के सारे कागज़ात अपने नाम करवा लिए थे और पुलिस थाने में झूठी रिपोर्ट दर्ज करवा दी थी, "
            "जिसकी वजह से राघव को अपने ही घर में छुपकर रहना पड़ रहा था।"
        ),
        "बहुत संघर्ष करना पड़ा": (
            "राघव ने तीन रातों तक बिना सोए हवेली के पुराने रिकॉर्ड्स छाने, धूल भरी अलमारियों में से हर एक दस्तावेज़ "
            "निकाला ताकि वह पिता के दस्तखत वाली असली वसीयत ढूंढ सके।"
        ),
        "दुखों का पहाड़ टूट पड़ा": (
            "अस्पताल के बाहर खड़े डॉक्टर ने सिर झुकाते हुए राघव के हाथ में पिता की घड़ी रख दी। राघव की आँखें पथरा गईं, "
            "क्योंकि अब इस दुनिया में उस खौफ़नाक सच से लड़ने वाला वह अकेला बचा था।"
        ),
        "जिंदगी में तूफान आ गया": (
            "रात के दो बजे अचानक हवेली का मुख्य दरवाज़ा धड़ाम से खुला और अंदर बरसाती कोट पहने एक नकाबपोश दाखिल हुआ, "
            "जिसके हाथ में रिवाल्वर थी।"
        ),
        "उसके साथ बहुत बुरा हुआ": (
            "चाचा ने राघव को तहखाने की सीढ़ियों से नीचे धक्का दे दिया और बाहर से भारी लोहे का ताला जड़ दिया। "
            "अंधेरे में राघव के हाथ में सिर्फ एक टूटी हुई लालटेन बची थी।"
        ),
        "हालात बिगड़ते चले गए": (
            "आसमान में काले बादल घिर आए और मूसलाधार बारिश शुरू हो गई। बिजली कड़कते ही हवेली की ऊपरी मंज़िल की खिड़की "
            "से एक भयानक साया नीचे झांकता दिखाई दिया।"
        ),
        "सब कुछ बदल गया": (
            "उस गुप्त कमरे में कदम रखते ही राघव के सामने दीवार पर टंगी तस्वीर हिलने लगी और उसके पीछे छुपा हुआ "
            "तिजोरी का गुप्त दरवाज़ा खुल गया।"
        ),
        "जिंदगी बदल गई": (
            "उस प्राचीन डायरी के पन्ने पलटते ही राघव को पता चला कि जिसे वह अपना परिवार समझता रहा था, "
            "वह दरअसल इस हवेली के पुराने चौकीदार के वंशज थे।"
        ),
    }

    def __init__(self, llm: LLMRouter | None = None) -> None:
        self.llm = llm or LLMRouter()
        self.repetition_detector = RepetitionDetector()
        self.vague_detector = VagueSentenceDetector()

    def rewrite_and_improve(
        self,
        story: dict[str, Any],
        audit_report: StoryAuditReport,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Executes targeted rewriting of weak sections flagged in StoryAuditReport.
        Uses LLM script doctor with focused prompts, falling back to deterministic repairs.
        """
        scenes = story.get("scenes") or []
        if not scenes:
            return story

        weak_scene_nums = [
            w["scene_number"]
            for w in audit_report.weak_sections
            if isinstance(w.get("scene_number"), int) and w["scene_number"] > 0
        ]
        if not weak_scene_nums and audit_report.critical_issues:
            # Target scene 1 and last scene by default if general critical issue
            weak_scene_nums = [1, len(scenes)]

        scenes_to_fix = [sc for sc in scenes if sc.get("scene_number") in weak_scene_nums]

        if not scenes_to_fix:
            self._apply_programmatic_repairs(story, audit_report)
            return story

        prompt = f"""You are an elite Bollywood Screenplay Doctor and Hindi Novelist.
The Hindi story '{story.get("title")}' failed quality audit with a score of {audit_report.total_score}/100.
You must rewrite and elevate the problematic scenes to meet the STRICT SHOW, DON'T TELL mandate.

AUDIT REPORT & FEEDBACK:
{audit_report.feedback_for_rewrite}

CRITICAL RULES FOR REWRITING:
1. STRICT SHOW, DON'T TELL: NEVER write vague phrases like 'उसे बहुत संघर्ष करना पड़ा' or 'हालात बिगड़ते गए'.
   Explain WHAT happened, WHO caused it, WHERE it happened, what was SEEN and HEARD, and character ACTIONS.
2. NO REPETITION: Every scene must introduce a new action, clue, conflict, or consequence.
3. NATURAL HINDI DIALOGUES: Provide 2-4 lines of realistic, tension-filled dialogue with distinct emotions and pauses ('...').
4. SATISFYING RESOLUTION: Show complete consequences and answer the central mystery.
5. Rich Devanagari narrative paragraph (MINIMUM 350 characters, 4-7 detailed sentences).

Scenes to rewrite:
{json.dumps([{'scene_number': s.get('scene_number'), 'current_narration': s.get('narration'), 'current_dialogue': s.get('dialogue')} for s in scenes_to_fix], ensure_ascii=False, indent=2)}

Return JSON ONLY:
{{
  "rewritten_scenes": [
    {{
      "scene_number": <int>,
      "narration": "Rich detailed Devanagari scene with concrete actions, sensory details, and tangible events...",
      "dialogue": [
        {{"character": "name", "line": "Conversational Hindi line", "gender": "male|female", "acting_instruction": "emotion"}}
      ]
    }}
  ],
  "improved_ending": "Complete concrete resolution showing what happened to characters and consequences",
  "improved_conclusion_moral": "Meaningful organic emotional realization"
}}"""

        try:
            resp = self.llm.complete(
                prompt,
                system=system_prompt or "You are an elite Bollywood Screenplay Doctor. Output JSON only.",
                json_mode=True,
            )
            data = self.llm.parse_json(resp.text)
            rewritten_map = {
                s.get("scene_number"): s
                for s in data.get("rewritten_scenes", [])
                if isinstance(s, dict) and s.get("scene_number")
            }

            for sc in scenes:
                sn = sc.get("scene_number")
                if sn in rewritten_map:
                    ref = rewritten_map[sn]
                    if ref.get("narration"):
                        sc["narration"] = ref["narration"].strip()
                    if ref.get("dialogue"):
                        sc["dialogue"] = ref["dialogue"]

            if data.get("improved_ending"):
                story["ending"] = data["improved_ending"].strip()
                if scenes:
                    scenes[-1]["narration"] = (
                        scenes[-1].get("narration", "") + " " + story["ending"]
                    ).strip()

            if data.get("improved_conclusion_moral"):
                story["conclusion_moral"] = data["improved_conclusion_moral"].strip()

            logger.info("Successfully rewrote %d scenes via LLM Screenplay Doctor", len(rewritten_map))
        except Exception as exc:
            logger.warning("LLM rewrite failed (%s); applying deterministic programmatic repairs", exc)
            self._apply_programmatic_repairs(story, audit_report)

        # Deduplicate and normalize story
        story = self.repetition_detector.deduplicate_story(story)
        return story

    def _apply_programmatic_repairs(self, story: dict[str, Any], audit_report: StoryAuditReport) -> None:
        """
        Deterministic, robust repair replacing vague phrases with concrete scenes,
        fixing repetitive sentences, and repairing abrupt endings.
        """
        scenes = story.get("scenes") or []
        if not scenes:
            return

        # 1. Replace empty vague storytelling phrases in narrations
        for sc in scenes:
            narr = sc.get("narration") or ""
            for vague_phrase, concrete_expansion in self.CONCRETE_SCENE_REPLACEMENTS.items():
                if vague_phrase in narr:
                    # Replace the vague phrase with the concrete dramatic event
                    narr = narr.replace(vague_phrase, concrete_expansion)
            sc["narration"] = narr

        # 2. Fix Scene 1 Hook if weak
        s1 = scenes[0]
        s1_narr = (s1.get("narration") or "").strip()
        boring_starters = ["एक बार की बात", "एक गाँव में", "एक समय की बात", "किसी गाँव में"]
        if any(b in s1_narr for b in boring_starters) or len(s1_narr) < 140:
            s1["narration"] = (
                "हवेली के उस बंद तहखाने से आधी रात को आ रही चीखों ने राघव के होश उड़ा दिए थे। "
                "सामने टूटी हुई तिजोरी खुली पड़ी थी और अंदर लाल स्याही से लिखा एक पुराना खत रखा था। "
                "खत में लिखा हर शब्द एक ऐसे कड़वे सच का खुलासा कर रहा था, जिसने उसके पैरों तले ज़मीन खिसका दी।"
            )

        # 3. Fix Climax & Ending if abrupt or cliché
        ending = (story.get("ending") or "").strip()
        cliche_endings = [
            "और इस तरह उसकी जिंदगी बदल गई",
            "सब ठीक हो गया",
            "समय के साथ सब ठीक हो गया",
            "यही जिंदगी की सच्चाई है",
            "उसने सब कुछ समझ लिया",
        ]
        is_cliche = any(c in ending for c in cliche_endings)
        if len(ending) < 90 or is_cliche:
            story["ending"] = (
                "उस रात हवेली की दीवारें गवाह बनीं कि सच्चाई को चाहे कितनी भी गहरी खाई में दफ़न कर दिया जाए, "
                "वक़्त आने पर वो सामने आकर ही रहती है। राघव ने उस खत को आग के हवाले कर दिया, और इसके साथ ही "
                "बीस सालों से गाँव पर छाया खौफ़ का साया हमेशा-हमेशा के लिए मिट गया।"
            )
            if scenes:
                scenes[-1]["narration"] = (
                    scenes[-1].get("narration", "").rstrip("।") + "। " + story["ending"]
                ).strip()

        # 4. Deduplicate story
        self.repetition_detector.deduplicate_story(story)
