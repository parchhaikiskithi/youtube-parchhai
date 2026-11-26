from __future__ import annotations

import logging
from typing import Any

from app.ai.providers import LLMRouter

logger = logging.getLogger(__name__)


class SEOGenerator:
    def __init__(self, llm: LLMRouter | None = None) -> None:
        self.llm = llm or LLMRouter()

    def generate(self, story: dict[str, Any], language: str) -> dict[str, Any]:
        research = story.get("research", {})
        hindi_rule = ""
        if language == "hi":
            hindi_rule = (
                "- recommended_title, title_variations, and description MUST be Hindi (Devanagari)\n"
                "- Titles must be curiosity-driven, terrifying, suspenseful and punchy (max 95 chars), without clickbait lies\n"
                "- tags MUST mix Hindi Devanagari and romanized Hindi horror/mystery keywords\n"
            )

        prompt = f"""Create YouTube Horror & Mystery SEO metadata for YouTube Channel: "परछाई किसकी थी?".
Language: {language}
Story Title: {story.get("title")}
Synopsis: {story.get("synopsis")}
Topic Research: {research}

SEO RULES FOR "परछाई किसकी थी?":
{hindi_rule}
- Channel Brand: "परछाई किसकी थी?"
- 5 curiosity-driven Hindi titles (max 95 chars), pick 1 recommended with highest CTR without misleading
- Description structure:
  Line 1-2: Terrifying cold hook + main horror keywords
  Line 3-5: Intriguing synopsis with unanswered mystery
  Line 6-8: Channel Call To Action: "अगर आपको 'परछाई किसकी थी?' की यह खौफनाक दास्तान पसंद आई तो वीडियो को लाइक करें और चैनल को सब्सक्राइब ज़रूर करें!"
  Line 9+: Chapter markers (00:00 शुरुआत, etc.)
- 15-25 highly relevant horror/mystery YouTube tags
- 5-8 hashtags (including #परछाईकिसकीथी #HindiHorrorStories #BhootKiKahani)
- Category ID: "24" (Entertainment)
- Playlist suggestion: "परछाई किसकी थी? - सच्ची भूतिया कहानियाँ"

JSON schema:
{{
  "main_topic": "...",
  "keywords": ["..."],
  "long_tail_keywords": ["..."],
  "title_variations": ["..."],
  "recommended_title": "...",
  "description": "...",
  "tags": ["..."],
  "hashtags": ["..."],
  "category_id": "24",
  "playlist_suggestion": "...",
  "chapters": [{{"time": "00:00", "title": "..."}}]
}}"""

        try:
            response = self.llm.complete(
                prompt,
                system="You are the YouTube SEO Director for the Hindi horror channel 'परछाई किसकी थी?'. Output valid JSON only.",
                json_mode=True,
            )
            return self.llm.parse_json(response.text)
        except Exception as exc:
            logger.warning("SEO generation LLM failed (%s). Using procedural horror SEO.", exc)
            title = story.get("title") or "हवेली का खौफ़नाक सच: परछाई किसकी थी?"
            clean_title = title.split("|")[0].split("-")[0].strip()
            return {
                "main_topic": title,
                "keywords": [
                    "परछाई किसकी थी",
                    "hindi horror stories",
                    "bhoot ki kahani",
                    "haunted haveli",
                    "horror stories in hindi",
                    "hindi mystery kahaniya",
                    "khaufnak sachi ghatna",
                ],
                "long_tail_keywords": [
                    "parchhai kiski thi horror story",
                    "rahasyamayi haveli ki kahani",
                    "real horror stories in hindi",
                    "sachi ghatna par aadharit kahani",
                ],
                "title_variations": [
                    f"{clean_title} | परछाई किसकी थी?",
                    f"आधी रात का खौफ़: {clean_title} | Hindi Horror Story",
                    f"{clean_title} - वो रात कोई नहीं भूला | सच्ची घटना",
                    f"उस कमरे में कौन था? {clean_title} | Mystery & Horror",
                    f"{clean_title} (रोंगटे खड़े कर देने वाली दास्तान)",
                ],
                "recommended_title": f"{clean_title} | परछाई किसकी थी? | Hindi Horror Story",
                "description": (
                    f"{clean_title} | परछाई किसकी थी?\n\n"
                    f"{story.get('synopsis', 'एक रोंगटे खड़े कर देने वाली खौफ़नाक और रहस्यमयी दास्तान। रात के अंधेरे में छुपे उस सच का खुलासा जिसे जानकर आपके होश उड़ जाएंगे।')}\n\n"
                    f"अगर आपको 'परछाई किसकी थी?' की यह रहस्यमयी और डरावनी कहानी पसंद आई हो, तो वीडियो को लाइक करें, दोस्तों के साथ शेयर करें और चैनल को सब्सक्राइब करना न भूलें!\n\n"
                    f"#परछाईकिसकीथी #HindiHorrorStories #BhootKiKahani #HorrorStoriesInHindi #HauntedHaveli #HindiKahaniya #SuspenseThriller"
                ),
                "tags": [
                    "परछाई किसकी थी",
                    "parchhai kiski thi",
                    "hindi horror stories",
                    "horror stories in hindi",
                    "bhoot ki kahani",
                    "haunted haveli",
                    "chudail ki kahani",
                    "khaufnak kahaniya",
                    "sachi ghatna",
                    "hindi mystery stories",
                    "horror story hindi",
                    "hindi kahaniya",
                    "bhootia kahani",
                    "darawani kahani",
                    "suspense horror",
                ],
                "hashtags": [
                    "#परछाईकिसकीथी",
                    "#HindiHorrorStories",
                    "#BhootKiKahani",
                    "#HorrorStoriesInHindi",
                    "#HauntedHaveli",
                    "#ParchhaiKiskiThi",
                ],
                "category_id": "24",
                "playlist_suggestion": "परछाई किसकी थी? - सर्वश्रेष्ठ खौफनाक कहानियाँ",
                "chapters": [
                    {"time": "00:00", "title": "खौफ़नाक शुरुआत"},
                    {"time": "02:30", "title": "पहला रहस्य और सुराग"},
                    {"time": "06:00", "title": "खतरे की दस्तक"},
                    {"time": "09:30", "title": "क्लाइमेक्स और चौंकाने वाला सच"},
                ],
            }

    def generate_full_package(
        self,
        story: dict[str, Any],
        language: str,
        *,
        category_id: str,
        topic_research: dict[str, Any],
        thumbnail_text: str | None = None,
    ) -> dict[str, Any]:
        base = self.generate(story, language)
        titles = base.get("title_variations") or []
        if len(titles) < 5:
            titles = titles + [base.get("recommended_title", "")] * (5 - len(titles))

        prompt = f"""You are the Creative Producer for YouTube Horror Channel: "परछाई किसकी थी?".
Category: {category_id}
Story Title: {story.get("title")}
Synopsis: {story.get("synopsis")}
Research: {topic_research}
Thumbnail Headline: {thumbnail_text or "परछाई किसकी थी?"}

Create social copy, pinned question, and SEO performance report in Hindi Devanagari.
JSON:
{{
  "title_selection_reason": "Why this title delivers maximum CTR for horror audience without being misleading",
  "promotional_caption": "2-3 sentence chilling teaser for social post",
  "pinned_comment": "One spine-chilling question to engage viewers in the comments",
  "hashtags_text": "#परछाईकिसकीथी #HindiHorrorStories ...",
  "seo_report": {{
    "primary_keyword": "...",
    "secondary_keywords": ["..."],
    "search_intent": "Audience looking for atmospheric Indian horror and mystery stories",
    "audience_fit": "Fans of Hindi supernatural thrillers and mystery storytelling",
    "competition_note": "Original cinematic atmosphere, high suspense and logical twist"
  }}
}}"""
        try:
            extra = self.llm.complete(
                prompt,
                system="Valid JSON only. Dedicated to horror channel 'परछाई किसकी थी?'.",
                json_mode=True,
            )
            extra_data = self.llm.parse_json(extra.text)
        except Exception as exc:
            logger.warning("Creative SEO LLM failed (%s). Using base horror SEO copy.", exc)
            extra_data = {
                "title_selection_reason": "High curiosity gap and dread for horror audience",
                "promotional_caption": f"क्या आप जानते हैं उस रात क्या हुआ था? सुनिए 'परछाई किसकी थी?' पर {story.get('title', '')} की यह रोंगटे खड़े कर देने वाली कहानी...",
                "pinned_comment": "अगर आप उस बंद हवेली में होते, तो क्या आप वो दरवाज़ा खोलने की हिम्मत करते? अपनी राय कमेंट में ज़रूर बताएं!",
                "hashtags_text": "#परछाईकिसकीथी #HindiHorrorStories #BhootKiKahani #HorrorStoriesInHindi",
                "seo_report": {
                    "primary_keyword": "परछाई किसकी थी",
                    "secondary_keywords": ["hindi horror stories", "haunted haveli", "bhoot ki kahani"],
                    "search_intent": "Audience seeking real-feeling Hindi horror and mystery",
                    "audience_fit": "Horror and suspense thriller viewers",
                    "competition_note": "No cheap jump scares; genuine dread and atmospheric storytelling",
                },
            }

        return {
            **base,
            "title_variations": titles[:5],
            "title_selection_reason": extra_data.get("title_selection_reason"),
            "promotional_caption": extra_data.get("promotional_caption"),
            "pinned_comment": extra_data.get("pinned_comment"),
            "hashtags_text": extra_data.get("hashtags_text"),
            "seo_report": extra_data.get("seo_report"),
            "selected_title": base.get("recommended_title"),
        }
