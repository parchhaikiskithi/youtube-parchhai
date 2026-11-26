from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from app.ai.providers import LLMRouter

logger = logging.getLogger(__name__)


@dataclass
class StoryBlueprint:
    concept: str
    main_character: dict[str, Any]
    supporting_characters: list[dict[str, Any]]
    character_relationships: list[str]
    character_motivations: dict[str, str]
    main_conflict: str
    antagonist_or_opposing_force: str
    important_events: list[dict[str, Any]]
    hidden_clues: list[dict[str, Any]]
    cause_and_effect_chain: list[str]
    major_twist: str
    climax: str
    final_resolution: str
    conclusion_moral: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StoryBlueprintGenerator:
    """
    Constructs a complete narrative blueprint before scene-by-scene script writing.
    Ensures that every event has a logical cause, every clue has an intentional setup
    and payoff, and no unintroduced elements appear at the climax.
    """

    def __init__(self, llm: LLMRouter | None = None) -> None:
        self.llm = llm or LLMRouter()

    def generate_blueprint(
        self,
        topic: dict[str, Any],
        category_name: str,
        genre: str = "horror",
        language: str = "hi",
    ) -> StoryBlueprint:
        """Generates a comprehensive blueprint using the LLM, with procedural fallback."""
        main_topic = topic.get("main_topic") or "हवेली का रहस्य"
        angle = topic.get("angle") or "बंद कमरे की परछाई"

        prompt = f"""You are an elite Bollywood Screenwriter, Hindi Novelist, and Story Architect.
Create a master scene-by-scene narrative plot blueprint for a cinematic Hindi story.

Genre: {genre}
Category: {category_name}
Topic: {main_topic}
Angle: {angle}

BLUEPRINT REQUIREMENTS:
1. NO VAGUE CONCEPTS: Define exactly who did what, when, where, and why.
2. CAUSE AND EFFECT: Every single major event must be the direct result of a previous action.
3. CLUES WITH PAYOFFS: Plant at least 2 tangible physical clues early that are solved in the twist or climax.
4. BELIEVABLE CHARACTERS: Strong clear motivations. Antagonist must have a clear human or supernatural origin.
5. NO RANDOM CLIMAX SURPRISES: Do NOT introduce a completely new character or unexplained secret at the climax.
6. SATISFYING RESOLUTION: Show what happens to every character and the final consequence.

Return valid JSON ONLY matching this schema:
{{
  "concept": "1-2 sentence core concept in Hindi Devanagari",
  "main_character": {{
    "name": "राघव",
    "gender": "male",
    "age_group": "27 years old",
    "profession": "खोजी पत्रकार",
    "physical_appearance": "Sharp Indian features, dark navy blue jacket",
    "personality": "साहसी और खोजी स्वभाव",
    "motivation": "अपने परिवार के लापता होने का सच जानना"
  }},
  "supporting_characters": [
    {{
      "name": "माया",
      "gender": "female",
      "age_group": "25 years old",
      "profession": "दस्तावेजी फिल्मकार",
      "personality": "तार्किक और संवेदनशील"
    }}
  ],
  "character_relationships": [
    "राघव और माया बचपन के दोस्त और खोजी साथी हैं"
  ],
  "character_motivations": {{
    "राघव": "पुरानी हवेली के बंद कमरे का सच दुनिया के सामने लाना",
    "माया": "अलौकिक दावों की वास्तविकता रिकॉर्ड करना"
  }},
  "main_conflict": "तीस साल पहले बंद किए गए कमरे का ताला आधी रात को अचानक अपने आप टूट जाता है",
  "antagonist_or_opposing_force": "हवेली के प्राचीन आईने में कैद साया जो परिवार के धोखे का प्रतिशोध चाहता है",
  "important_events": [
    {{
      "event_number": 1,
      "setup": "राघव और माया रात 12 बजे हवेली के दालान में पहुंचते हैं",
      "event": "दीवार पर टंगी बंद घड़ी उलटी दिशा में चलने लगती है और मोमबत्तियां अपने आप जल उठती हैं",
      "action": "राघव टॉर्च से फर्श पर कदमों के निशान तलाशता है",
      "reaction": "माया कांपते हाथों से कैमरा चालू करती है",
      "consequence": "उन्हें पहली मंजिल के सील किए गए कमरे का रास्ता मिलता है"
    }}
  ],
  "hidden_clues": [
    {{
      "clue": "चौखट में छिपा पीतल का ताबीज और लाल डायरी",
      "planted_at": "शुरुआती दृश्य (कमरे के बाहर)",
      "payoff_at": "चरमोत्कर्ष (आईने के सामने रहस्योद्घाटन)"
    }}
  ],
  "cause_and_effect_chain": [
    "कमरा खुला -> काला धुआं निकला -> पुरानी डायरी मिली -> सच सामने आया"
  ],
  "major_twist": "आईने में दिखने वाली परछाई किसी अजनबी की नहीं, बल्कि राघव के अपने परिवार के पूर्वज की है",
  "climax": "राघव आईने के सामने खड़े होकर उस डायरी के आखिरी पन्नों को पढ़कर सच स्वीकार करता है",
  "final_resolution": "राघव उस डायरी को नष्ट कर हवेली से बाहर आता है, जहां सुबह की पहली किरण फैल रही है",
  "conclusion_moral": "अतीत के सच को चाहे कितनी भी गहरी खामोशी में दफन कर दिया जाए, उसकी परछाई पीछा नहीं छोड़ती"
}}"""

        try:
            resp = self.llm.complete(
                prompt,
                system="You are an uncompromising master screenwriter. Output valid JSON only.",
                json_mode=True,
            )
            data = self.llm.parse_json(resp.text)
            return self._build_blueprint_from_dict(data)
        except Exception as exc:
            logger.warning("LLM Blueprint generation failed (%s); building procedural blueprint", exc)
            return self.generate_procedural_blueprint(main_topic, category_name)

    def _build_blueprint_from_dict(self, data: dict[str, Any]) -> StoryBlueprint:
        return StoryBlueprint(
            concept=data.get("concept") or "रहस्यमयी घटना का पर्दाफाश",
            main_character=data.get("main_character") or {
                "name": "राघव",
                "gender": "male",
                "age_group": "27 years old",
                "profession": "खोजी पत्रकार",
                "physical_appearance": "Sharp Indian features, dark navy blue jacket",
                "personality": "साहसी और खोजी",
                "motivation": "सच्चाई तक पहुंचना",
            },
            supporting_characters=data.get("supporting_characters") or [
                {"name": "माया", "gender": "female", "age_group": "25 years old", "profession": "दस्तावेजी साथी"}
            ],
            character_relationships=data.get("character_relationships") or ["राघव और माया खोजी साथी हैं"],
            character_motivations=data.get("character_motivations") or {
                "राघव": "हवेली के बंद कमरे का सच जानना",
                "माया": "साक्ष्यों को कैमरे में कैद करना",
            },
            main_conflict=data.get("main_conflict") or "आधी रात को बंद हवेली में अलौकिक परछाई का प्रकट होना",
            antagonist_or_opposing_force=data.get("antagonist_or_opposing_force") or "हवेली के आईने में कैद प्राचीन साया",
            important_events=data.get("important_events") or [],
            hidden_clues=data.get("hidden_clues") or [],
            cause_and_effect_chain=data.get("cause_and_effect_chain") or [],
            major_twist=data.get("major_twist") or "परछाई का मुख्य पात्र के अपने अतीत से सीधा संबंध होना",
            climax=data.get("climax") or "आईने के सामने सत्य का प्रत्यक्ष सामना",
            final_resolution=data.get("final_resolution") or "सच्चाई सामने आने पर हवेली का खौफ शांत होना",
            conclusion_moral=data.get("conclusion_moral") or "सच्चाई को चाहे जितनी गहराई में छुपाया जाए, वह सामने आकर ही रहती है",
        )

    def generate_procedural_blueprint(self, title: str, category_name: str) -> StoryBlueprint:
        """High-grade procedural fallback blueprint ensuring tight cause-and-effect continuity."""
        return StoryBlueprint(
            concept=f"{category_name} के तहत {title} की रहस्यमयी गुत्थी को सुलझाने की रोमांचक और खौफनाक दास्तान।",
            main_character={
                "name": "राघव",
                "gender": "male",
                "age_group": "27 years old",
                "profession": "खोजी पत्रकार",
                "physical_appearance": "Sharp Indian features, clean shaven, intense piercing eyes, dark navy blue jacket",
                "personality": "साहसी, बारीकियों पर नज़र रखने वाला, तार्किक",
                "motivation": "तीस साल पहले इस हवेली से लापता हुए परिवार का सच उजागर करना",
            },
            supporting_characters=[
                {
                    "name": "माया",
                    "gender": "female",
                    "age_group": "25 years old",
                    "profession": "दस्तावेजी फिल्मकार",
                    "physical_appearance": "Expressive large dark eyes, olive green outdoor jacket, camera around neck",
                    "personality": "संवेदनशील, अलौकिक माहौल को तुरंत भांपने वाली",
                    "motivation": "घटनास्थल की हर हरकत को कैमरे में कैद कर सच का दस्तावेज़ बनाना",
                }
            ],
            character_relationships=["राघव और माया कई वर्षों से साथ काम करने वाले खोजी पत्रकार साथी हैं"],
            character_motivations={
                "राघव": "अपने परिवार से जुड़े इस अनसुलझे रहस्य की तह तक पहुँचना",
                "माया": "खतरे के बावजूद राघव का साथ देना और सच्चाई को रिकॉर्ड करना",
            },
            main_conflict="तीस साल से सील बंद कमरे से अचानक लाल रोशनी और चीखें सुनाई देना, जबकि कमरे में कोई दाखिल नहीं हुआ था।",
            antagonist_or_opposing_force="प्राचीन हवेली के विशाल आदमकद आईने में बंधी अलौकिक शक्ति जो अपने साथ हुए विश्वासघात का हिसाब चाहती है।",
            important_events=[
                {
                    "event_number": 1,
                    "setup": "राघव और माया रात के 12 बजे हवेली के मुख्य दालान में प्रवेश करते हैं।",
                    "event": "दीवार पर टंगी पुरानी पेंडुलम घड़ी उलटी दिशा में चलने लगती है और मोमबत्तियां बिना छुए नीली लौ के साथ जल उठती हैं।",
                    "action": "राघव टॉर्च से सीढ़ियों पर ताज़ा पैरों के निशान देखता है।",
                    "reaction": "माया कांपते हाथों से रिकॉर्डिंग शुरू करती है।",
                    "consequence": "निशान उन्हें पहली मंजिल के भारी लोहे की ज़ंजीरों से बंधे कमरे तक ले जाते हैं।",
                },
                {
                    "event_number": 2,
                    "setup": "पहली मंजिल के बंद कमरे की चौखट पर अजीबोगरीब लाल निशान दिखाई देते हैं।",
                    "event": "दीवार के गुप्त आले से एक पुरानी लाल डायरी और पीतल की चाबी मिलती है।",
                    "action": "राघव डायरी खोलकर तीस साल पुरानी आखिरी प्रविष्टि पढ़ता है।",
                    "reaction": "दोनों को पता चलता है कि परिवार भूत से नहीं, अपने ही धोखे के डर से बंद हुआ था।",
                    "consequence": "कमरे का भारी दरवाज़ा ज़ंजीरें तोड़कर अचानक अपने आप खुल जाता है।",
                },
                {
                    "event_number": 3,
                    "setup": "कमरे के अंदर सिर्फ एक विशाल प्राचीन आदमकद आईना रखा है।",
                    "event": "आईने में राघव की परछाई उसके साथ नहीं हिलती, बल्कि उसके पीछे एक साया मुस्कुराता है।",
                    "action": "राघव डायरी का वो पन्ना देखता है जिसमें उसके अपने बचपन की तस्वीर है।",
                    "reaction": "राघव समझ जाता है कि वह उसी परिवार का खोया हुआ वारिस है।",
                    "consequence": "साया शांत हो जाता है क्योंकि वारिस ने सच को स्वीकार कर लिया।",
                },
            ],
            hidden_clues=[
                {
                    "clue": "दीवार के गुप्त आले में रखी लाल डायरी और पीतल की पुरानी चाबी",
                    "planted_at": "शुरुआती दृश्य में सीढ़ियों के पास",
                    "payoff_at": "चरमोत्कर्ष में राघव के वंशज होने का प्रमाण",
                },
                {
                    "clue": "आईने की चौखट पर खुदा वही निशान जो राघव के हाथ के ताबीज पर है",
                    "planted_at": "कमरे में प्रवेश करते वक्त",
                    "payoff_at": "आईने के सामने खड़े होने पर पहचान की पुष्टि",
                },
            ],
            cause_and_effect_chain=[
                "हवेली में प्रवेश -> कदमों के निशान -> पहली मंजिल का सीलबंद कमरा -> गुप्त डायरी की खोज -> आईने का रहस्योद्घाटन -> सत्य की स्वीकृति और शांति",
            ],
            major_twist="कमरे में दिखने वाला साया कोई बाहरी चुड़ैल या शैतान नहीं, बल्कि राघव के अपने परिवार का संरक्षक साया था जो केवल सच का इंतज़ार कर रहा था।",
            climax="राघव आईने के ठीक सामने खड़े होकर अपने परिवार के अतीत का खौफनाक सच स्वीकार करता है और डायरी की आखिरी कसम पूरी करता है।",
            final_resolution="हवेली से बरसों पुराना अंधेरा और खौफ छंट जाता है; राघव और माया सुबह के उजाले में हवेली से सुरक्षित बाहर निकलते हैं।",
            conclusion_moral="अतीत के सच को चाहे कितनी भी गहरी खामोशी में दफन कर दिया जाए, उसकी परछाई कभी पीछा नहीं छोड़ती।",
        )
