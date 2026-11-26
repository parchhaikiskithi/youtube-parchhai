from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LogicFlaw:
    flaw_type: str  # "unintroduced_character", "unresolved_clue", "abrupt_ending", "cliche_ending", "broken_continuity"
    scene_number: int
    description: str
    severity: str = "critical"  # "critical" or "warning"


@dataclass
class LogicValidationReport:
    passed: bool
    penalty_score: int
    flaws: list[LogicFlaw] = field(default_factory=list)
    critical_issues: list[str] = field(default_factory=list)
    unresolved_clues: list[str] = field(default_factory=list)
    character_flaws: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "penalty_score": self.penalty_score,
            "critical_issues": self.critical_issues,
            "flaws": [
                {
                    "flaw_type": f.flaw_type,
                    "scene_number": f.scene_number,
                    "description": f.description,
                    "severity": f.severity,
                }
                for f in self.flaws
            ],
            "unresolved_clues": self.unresolved_clues,
            "character_flaws": self.character_flaws,
        }


class StoryLogicValidator:
    """
    Validates narrative cause-and-effect continuity, character consistency,
    clue setup/payoff, and resolution completeness for Hindi stories.
    """

    KNOWN_CLUE_KEYWORDS = [
        "खत", "चिट्ठी", "डायरी", "चाबी", "ताला", "आईना", "तिजोरी", "निशान",
        "फोटो", "तस्वीर", "रिकॉर्डिंग", "वसीयत", "कंकाल", "ज़ंजीर", "दस्तावेज़",
    ]

    BANNED_ENDING_CLICHES = [
        "और इस तरह उसकी जिंदगी बदल गई",
        "समय के साथ सब ठीक हो गया",
        "सब ठीक हो गया",
        "उसने सब कुछ समझ लिया",
        "यही जिंदगी की सच्चाई है",
        "जिंदगी सामान्य हो गई",
    ]

    ALLOWED_GENERIC_SPEAKERS = {
        "सूत्रधार", "वॉइसओवर", "Narrator", "Voiceover", "पात्र", "अजनबी", "भीड़",
        "दुकानदार", "पुजारी", "चौकीदार", "तांत्रिक", "आवाज़", "परछाई",
    }

    def validate(self, story: dict[str, Any]) -> LogicValidationReport:
        scenes = story.get("scenes") or []
        flaws: list[LogicFlaw] = []
        critical_issues: list[str] = []
        character_flaws: list[str] = []
        unresolved_clues: list[str] = []

        total_scenes = len(scenes)
        is_short = bool(story.get("is_short"))

        # 1. Minimum Scene Count Check
        min_scenes = 4 if is_short else 7
        if total_scenes < min_scenes:
            msg = f"Insufficient scene progression: {total_scenes} scenes (expected minimum {min_scenes})"
            flaws.append(LogicFlaw("broken_continuity", 0, msg, "critical"))
            critical_issues.append(msg)

        # 2. Character Consistency Validation
        defined_chars = set()
        for c in (story.get("characters") or []):
            if isinstance(c, dict) and c.get("name"):
                defined_chars.add(c["name"].strip())
        for cr in (story.get("character_references") or []):
            if isinstance(cr, dict) and cr.get("name"):
                defined_chars.add(cr["name"].strip())

        dialogue_speakers_by_scene: dict[int, list[str]] = {}
        for sc in scenes:
            sn = sc.get("scene_number", 0)
            speakers = []
            for d in sc.get("dialogue") or []:
                if isinstance(d, dict) and d.get("character"):
                    speakers.append(d["character"].strip())
            dialogue_speakers_by_scene[sn] = speakers

        all_speakers = {s for spk_list in dialogue_speakers_by_scene.values() for s in spk_list}
        unknown_speakers = all_speakers - defined_chars - self.ALLOWED_GENERIC_SPEAKERS

        if len(unknown_speakers) > 2:
            msg = f"Undefined characters speaking in dialogue without introduction: {list(unknown_speakers)}"
            flaws.append(LogicFlaw("unintroduced_character", 0, msg, "warning"))
            character_flaws.append(msg)

        # Check if climax (last 3 scenes) introduces a brand-new unexplained major character
        if total_scenes >= 5:
            early_speakers = {
                s for sn, spks in dialogue_speakers_by_scene.items() if sn <= total_scenes - 3 for s in spks
            }
            climax_speakers = {
                s for sn, spks in dialogue_speakers_by_scene.items() if sn > total_scenes - 3 for s in spks
            }
            brand_new_climax = (climax_speakers - early_speakers) - defined_chars - self.ALLOWED_GENERIC_SPEAKERS
            for bnc in brand_new_climax:
                msg = f"New character '{bnc}' suddenly appears in climax scenes without prior introduction (Deus Ex Machina)"
                flaws.append(LogicFlaw("unintroduced_character", total_scenes - 2, msg, "critical"))
                critical_issues.append(msg)

        # 3. Clue Setup and Payoff Tracking
        planted_clues: dict[str, int] = {}
        resolved_clues: set[str] = set()

        for sc in scenes:
            sn = sc.get("scene_number", 0)
            narr = sc.get("narration") or ""
            diags = " ".join((d.get("line") or "") for d in sc.get("dialogue") or [])
            full_scene_text = narr + " " + diags

            for clue in self.KNOWN_CLUE_KEYWORDS:
                if clue in full_scene_text:
                    if clue not in planted_clues:
                        planted_clues[clue] = sn
                    # If mentioned in second half or climax, count as addressed/resolved
                    if sn > total_scenes // 2:
                        resolved_clues.add(clue)

        # Major clues planted early (in first 3 scenes) that never appear in second half
        for clue, planted_sn in planted_clues.items():
            if planted_sn <= 3 and clue not in resolved_clues and total_scenes >= 6:
                unresolved_clues.append(clue)
                flaws.append(
                    LogicFlaw(
                        "unresolved_clue",
                        planted_sn,
                        f"Major clue '{clue}' introduced in Scene {planted_sn} is never explained or resolved",
                        "warning",
                    )
                )

        # 4. Climax and Ending Validation
        if scenes:
            ending = (story.get("ending") or "").strip()
            last_narr = (scenes[-1].get("narration") or "").strip()
            full_end = ending + " " + last_narr

            # Cliche endings
            for c in self.BANNED_ENDING_CLICHES:
                if c in full_end:
                    msg = f"Story ending uses forbidden cliché: '{c}'"
                    flaws.append(LogicFlaw("cliche_ending", total_scenes, msg, "critical"))
                    critical_issues.append(msg)

            # Abrupt ending check
            if len(full_end.strip()) < 90:
                msg = "Story ending is too abrupt (< 90 characters) and fails to resolve consequences"
                flaws.append(LogicFlaw("abrupt_ending", total_scenes, msg, "critical"))
                critical_issues.append(msg)

        # Calculate penalty and pass status
        penalty = 0
        for f in flaws:
            penalty += 8 if f.severity == "critical" else 3
        penalty = min(penalty, 40)

        passed = (len(critical_issues) == 0) and (penalty < 20)

        return LogicValidationReport(
            passed=passed,
            penalty_score=penalty,
            flaws=flaws,
            critical_issues=critical_issues,
            unresolved_clues=unresolved_clues,
            character_flaws=character_flaws,
        )
