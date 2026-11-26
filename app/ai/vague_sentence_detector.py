from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VagueSentenceMatch:
    scene_number: int
    vague_phrase: str
    sentence: str
    context: str
    diagnostic_questions: list[str]
    severity: str  # "critical" if unsupported summary, "warning" if minor


@dataclass
class VagueReport:
    has_vague_storytelling: bool
    vague_count: int
    penalty_score: int
    critical_issues: list[str] = field(default_factory=list)
    matches: list[VagueSentenceMatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_vague_storytelling": self.has_vague_storytelling,
            "vague_count": self.vague_count,
            "penalty_score": self.penalty_score,
            "critical_issues": self.critical_issues,
            "matches": [
                {
                    "scene_number": m.scene_number,
                    "vague_phrase": m.vague_phrase,
                    "sentence": m.sentence,
                    "diagnostic_questions": m.diagnostic_questions,
                    "severity": m.severity,
                }
                for m in self.matches
            ],
        }


class VagueSentenceDetector:
    """
    Detects empty, vague, telling-instead-of-showing storytelling in Hindi text.
    Enforces the 'SHOW WHAT ACTUALLY HAPPENED' rule by verifying that every mentioned
    hardship, betrayal, discovery, or emotion is backed by concrete sensory details,
    named participants, tangible actions, and consequences.
    """

    # Vague Hindi phrases explicitly banned unless immediately followed by concrete events
    BANNED_EMPTY_PATTERNS = [
        (r"मुश्किलों का सामना करना पड़ा", "मुश्किलों का सामना करना पड़ा"),
        (r"परेशानियां बढ़ती चली गईं", "परेशानियां बढ़ती चली गईं"),
        (r"हालात बहुत खराब हो गए", "हालात बहुत खराब हो गए"),
        (r"समस्याओं से गुजरना पड़ा", "समस्याओं से गुजरना पड़ा"),
        (r"बहुत संघर्ष करना पड़ा", "बहुत संघर्ष करना पड़ा"),
        (r"दुखों का पहाड़ टूट पड़ा", "दुखों का पहाड़ टूट पड़ा"),
        (r"जिंदगी में तूफान आ गया", "जिंदगी में तूफान आ गया"),
        (r"उसने बहुत कुछ सहा", "उसने बहुत कुछ सहा"),
        (r"उसके साथ बहुत बुरा हुआ", "उसके साथ बहुत बुरा हुआ"),
        (r"हालात बिगड़ते चले गए", "हालात बिगड़ते चले गए"),
        (r"सब कुछ बदल गया", "सब कुछ बदल गया"),
        (r"सब कुछ बदल चुका था", "सब कुछ बदल चुका था"),
        (r"जिंदगी बदल गई", "जिंदगी बदल गई"),
        (r"कठिनाइयों का सामना किया", "कठिनाइयों का सामना किया"),
        (r"एक के बाद एक मुसीबतें", "एक के बाद एक मुसीबतें आती रहीं"),
        (r"किस्मत ने उसका साथ छोड़ दिया", "किस्मत ने उसका साथ छोड़ दिया"),
        (r"समय बीतता गया", "समय बीतता गया"),
        (r"परेशानियां झेलनी पड़ीं", "परेशानियां झेलनी पड़ीं"),
        (r"और इस तरह उसकी जिंदगी बदल गई", "और इस तरह उसकी जिंदगी बदल गई"),
        (r"उसने सब कुछ समझ लिया", "उसने सब कुछ समझ लिया"),
        (r"समय के साथ सब ठीक हो गया", "समय के साथ सब ठीक हो गया"),
        (r"यही जिंदगी की सच्चाई है", "यही जिंदगी की सच्चाई है"),
        (r"दुनिया पहले जैसी नहीं रही", "दुनिया पहले जैसी नहीं रही"),
        (r"बड़ा बदलाव आ गया", "बड़ा बदलाव आ गया"),
    ]

    CONCRETE_MARKERS = [
        # Named entities / relationships
        r"(?:चाचा|मामा|पिता|माँ|भाई|दोस्त|साहब|राघव|अमित|रोहित|माया|नेहा|सुरेश|इंस्पेक्टर|डॉक्टर)",
        # Concrete physical objects
        r"(?:कागज़|दस्तावेज़|चाबी|तिजोरी|खत|चिट्ठी|आईना|दरवाज़ा|खिड़की|ताला|बंदूक|खून|डायरी|फ़ोन|गाड़ी|दीवार|सीढ़ियां|तस्वीर|कमरा|बक्सा|रुपये|लाख)",
        # Specific dialogue/action verbs
        r"(?:कहा|पूछा|चिल्लाया|हाथ कांपने|दस्तखत|हस्ताक्षर|गिरा|खोला|तोड़ा|दिखाया|पहुंचा|देखा|सुना|चीख|कदम)",
    ]

    SHOW_DONT_TELL_QUESTIONS = [
        "What exactly happened at this moment?",
        "Who caused this specific event or conflict?",
        "Where and when did this take place?",
        "What concrete physical objects or documents were seen?",
        "What specific words were spoken between the characters?",
        "What immediate consequences occurred?",
    ]

    @classmethod
    def is_concrete_scene(cls, text: str) -> bool:
        """Checks if a paragraph has tangible concrete markers (entities, objects, specific actions)."""
        marker_count = 0
        for pat in cls.CONCRETE_MARKERS:
            if re.search(pat, text):
                marker_count += 1
        return marker_count >= 2

    def check_story(self, story: dict[str, Any]) -> VagueReport:
        """
        Audits all scenes and top-level ending fields for vague storytelling phrases.
        Flags critical issues when a scene asserts an emotional/plot development
        without concrete physical scene evidence.
        """
        scenes = story.get("scenes") or []
        matches: list[VagueSentenceMatch] = []
        critical_issues: list[str] = []

        for sc in scenes:
            sn = sc.get("scene_number", 0)
            narr = (sc.get("narration") or "").strip()
            if not narr:
                continue

            # Split narration into individual sentences
            parts = re.split(r"[।?!;\n]+", narr)
            sentences = [p.strip() for p in parts if len(p.strip()) > 5]

            scene_is_concrete = self.is_concrete_scene(narr)

            for sent in sentences:
                for regex_pat, phrase_name in self.BANNED_EMPTY_PATTERNS:
                    if re.search(regex_pat, sent):
                        # If the sentence uses a banned trope and the paragraph lacks concrete details,
                        # or if the sentence stands alone as an abstract assertion:
                        is_unsupported = not scene_is_concrete or len(sentences) <= 2
                        severity = "critical" if is_unsupported else "warning"

                        match = VagueSentenceMatch(
                            scene_number=sn,
                            vague_phrase=phrase_name,
                            sentence=sent,
                            context=narr[:140],
                            diagnostic_questions=self.SHOW_DONT_TELL_QUESTIONS,
                            severity=severity,
                        )
                        matches.append(match)

                        if severity == "critical":
                            critical_issues.append(
                                f"Scene {sn} uses empty summary phrase '{phrase_name}' without showing concrete events"
                            )
                        break

        # Check ending field
        ending = (story.get("ending") or "").strip()
        cliche_endings = [
            "और इस तरह उसकी जिंदगी बदल गई",
            "सब ठीक हो गया",
            "समय के साथ सब ठीक हो गया",
            "यही जिंदगी की सच्चाई है",
            "उसने सब कुछ समझ लिया",
        ]
        for c in cliche_endings:
            if c in ending:
                matches.append(
                    VagueSentenceMatch(
                        scene_number=len(scenes),
                        vague_phrase=c,
                        sentence=ending,
                        context=ending,
                        diagnostic_questions=[
                            "How was the central conflict resolved?",
                            "What happened to each character?",
                            "What were the final real-world consequences?",
                        ],
                        severity="critical",
                    )
                )
                critical_issues.append(f"Ending relies on vague cliché '{c}' instead of showing final resolution")

        # Penalty score calculation
        penalty = 0
        for m in matches:
            if m.severity == "critical":
                penalty += 6
            else:
                penalty += 2

        penalty = min(penalty, 35)
        has_vague = len(matches) > 0

        return VagueReport(
            has_vague_storytelling=has_vague,
            vague_count=len(matches),
            penalty_score=penalty,
            critical_issues=critical_issues,
            matches=matches,
        )
