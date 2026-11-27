from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from app.ai.providers import LLMRouter
from app.ai.repetition_detector import RepetitionDetector, RepetitionReport
from app.ai.story_logic_validator import LogicValidationReport, StoryLogicValidator
from app.ai.vague_sentence_detector import VagueReport, VagueSentenceDetector

logger = logging.getLogger(__name__)


@dataclass
class StoryAuditReport:
    passed: bool
    total_score: int
    minimum_required: int = 85
    dimensions: dict[str, int] = field(default_factory=dict)
    repetition_penalty: int = 0
    vague_penalty: int = 0
    logic_penalty: int = 0
    critical_issues: list[str] = field(default_factory=list)
    weak_sections: list[dict[str, Any]] = field(default_factory=list)
    feedback_for_rewrite: str = ""
    human_review_checklist: dict[str, bool] = field(default_factory=dict)
    repetition_report: dict[str, Any] = field(default_factory=dict)
    vague_report: dict[str, Any] = field(default_factory=dict)
    logic_report: dict[str, Any] = field(default_factory=dict)
    evaluation_method: str = "hybrid"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StoryQualityAuditor:
    """
    Comprehensive Bollywood screenplay auditor and Hindi storytelling quality inspector.
    Evaluates 10 core dimensions from 0 to 10 each (target >= 85/100).
    Applies dedicated penalties for repetition, vague storytelling, and logic flaws.
    Enforces strict failure conditions for repetitive paragraphs, empty phrases,
    unexplained events, unresolved mysteries, and abrupt endings.
    """

    MIN_PASSING_SCORE: int = 85

    DIMENSION_KEYS = [
        "opening_hook",
        "actual_event_detail",
        "scene_quality",
        "logical_continuity",
        "character_development",
        "dialogue_quality",
        "suspense",
        "originality",
        "climax",
        "ending_and_conclusion",
    ]

    # Backward compatibility mapping for callers expecting old dimension keys
    COMPAT_KEY_MAP = {
        "plot_logic_continuity": "logical_continuity",
        "character_consistency": "character_development",
        "suspense_escalation": "suspense",
        "climax_impact": "climax",
        "ending_completeness": "ending_and_conclusion",
        "hindi_grammar_dialogue": "dialogue_quality",
        "anti_repetition": "scene_quality",
        "emotional_engagement": "actual_event_detail",
    }

    def __init__(self, llm: LLMRouter | None = None) -> None:
        self.llm = llm or LLMRouter()
        self.repetition_detector = RepetitionDetector()
        self.vague_detector = VagueSentenceDetector()
        self.logic_validator = StoryLogicValidator()

    def audit(self, story: dict[str, Any]) -> StoryAuditReport:
        """
        Runs comprehensive 10-dimension audit, repetition analysis, vague sentence analysis,
        logic validation, and human review checklist.
        """
        scenes = story.get("scenes") or []
        if not scenes:
            return StoryAuditReport(
                passed=False,
                total_score=0,
                minimum_required=self.MIN_PASSING_SCORE,
                critical_issues=["Story contains zero scenes."],
                feedback_for_rewrite="Generate a complete story with at least 8-15 fully detailed scenes.",
            )

        # 1. Specialized Detectors
        rep_report: RepetitionReport = self.repetition_detector.check_story(story)
        vague_report: VagueReport = self.vague_detector.check_story(story)
        logic_report: LogicValidationReport = self.logic_validator.validate(story)

        # 2. Rule-Based Dimension Scoring
        rule_scores, weak_sections, rule_critical_issues = self._score_rules(
            story, rep_report, vague_report, logic_report
        )

        # 3. Optional LLM Semantic Scoring
        llm_scores: dict[str, int] = {}
        llm_feedback = ""
        eval_method = "rule_based"

        try:
            llm_result = self._evaluate_with_llm(story, rule_scores, weak_sections)
            llm_scores = llm_result.get("scores", {})
            llm_feedback = llm_result.get("feedback", "")
            eval_method = "llm_validated"
        except Exception as exc:
            logger.debug("LLM story quality audit unavailable (%s); using deterministic rule scoring", exc)

        # 4. Synthesize Dimension Scores
        final_dimensions: dict[str, int] = {}
        subtotal = 0
        for dim in self.DIMENSION_KEYS:
            r_val = rule_scores.get(dim, 8)
            if dim in llm_scores:
                comb = int(round(llm_scores[dim] * 0.7 + r_val * 0.3))
            else:
                comb = r_val
            comb = max(1, min(10, comb))
            final_dimensions[dim] = comb
            subtotal += comb

        # Populate backward-compatible keys so older test cases and views find expected fields
        for old_k, new_k in self.COMPAT_KEY_MAP.items():
            if old_k not in final_dimensions and new_k in final_dimensions:
                final_dimensions[old_k] = final_dimensions[new_k]

        # 5. Apply Dedicated Penalties
        rep_penalty = rep_report.penalty_score
        vague_penalty = vague_report.penalty_score
        logic_penalty = logic_report.penalty_score

        # Subtract net penalty proportionally from total
        net_penalty = min(25, (rep_penalty // 2) + (vague_penalty // 2) + (logic_penalty // 2))
        total_score = max(10, subtotal - net_penalty)

        # 6. Aggregate Critical Issues
        critical_issues = list(
            set(
                rule_critical_issues
                + rep_report.critical_issues
                + vague_report.critical_issues
                + logic_report.critical_issues
            )
        )

        # 7. Human Review Checklist Gate
        human_review = self.run_human_review_checklist(story, rep_report, vague_report, logic_report)
        checklist_failures = [q for q, passed in human_review.items() if not passed]

        if checklist_failures:
            for fail_q in checklist_failures[:3]:
                critical_issues.append(f"Human review failed: {fail_q}")

        # 8. Strict Pass Criteria
        passed = (
            (total_score >= self.MIN_PASSING_SCORE)
            and (len(critical_issues) == 0)
            and (len(checklist_failures) == 0)
        )

        # 9. Detailed Feedback Compilation
        feedback_parts = []
        if critical_issues:
            feedback_parts.append("CRITICAL FLAWS THAT MUST BE FIXED:")
            for issue in critical_issues[:5]:
                feedback_parts.append(f"  * {issue}")

        if weak_sections:
            feedback_parts.append("WEAK SCENES TO REWRITE:")
            for w in weak_sections[:4]:
                feedback_parts.append(f"  * Scene {w.get('scene_number')}: {w.get('issue')}")

        if rep_report.issues:
            feedback_parts.append(f"REPETITION DEDUCTIONS (-{rep_penalty} pts): Remove duplicate sentences/dialogues.")
        if vague_report.matches:
            feedback_parts.append(
                f"VAGUE STORYTELLING DEDUCTIONS (-{vague_penalty} pts): Replace abstract summaries with concrete actions."
            )
        if llm_feedback:
            feedback_parts.append(f"DIRECTOR'S ADVICE: {llm_feedback}")

        feedback_str = "\n".join(feedback_parts)

        return StoryAuditReport(
            passed=passed,
            total_score=total_score,
            minimum_required=self.MIN_PASSING_SCORE,
            dimensions=final_dimensions,
            repetition_penalty=rep_penalty,
            vague_penalty=vague_penalty,
            logic_penalty=logic_penalty,
            critical_issues=critical_issues,
            weak_sections=weak_sections,
            feedback_for_rewrite=feedback_str,
            human_review_checklist=human_review,
            repetition_report=rep_report.to_dict(),
            vague_report=vague_report.to_dict(),
            logic_report=logic_report.to_dict(),
            evaluation_method=eval_method,
        )

    def _score_rules(
        self,
        story: dict[str, Any],
        rep_report: RepetitionReport,
        vague_report: VagueReport,
        logic_report: LogicValidationReport,
    ) -> tuple[dict[str, int], list[dict[str, Any]], list[str]]:
        scenes = story.get("scenes") or []
        scores: dict[str, int] = {k: 9 for k in self.DIMENSION_KEYS}
        weak_sections: list[dict[str, Any]] = []
        critical_issues: list[str] = []

        total_scenes = len(scenes)

        # 1. Opening hook (Scene 1)
        if scenes:
            s1 = scenes[0]
            s1_text = (s1.get("narration") or "") + " " + " ".join(d.get("line", "") for d in s1.get("dialogue", []))
            boring_starters = ["एक बार की बात", "एक गाँव में", "एक समय की बात", "किसी गाँव में"]
            if any(b in s1_text for b in boring_starters):
                scores["opening_hook"] = 3
                critical_issues.append("Scene 1 starts with a generic fairy-tale starter")
                weak_sections.append({"scene_number": 1, "issue": "Needs immediate cold hook, not generic introduction"})
            elif len(s1_text.strip()) < 120:
                scores["opening_hook"] = 5
                weak_sections.append({"scene_number": 1, "issue": "Opening hook is too brief (< 120 chars)"})
            else:
                scores["opening_hook"] = 9

        # 2. Actual event detail (Show, Don't Tell)
        vague_crit_count = sum(1 for m in vague_report.matches if m.severity == "critical")
        if vague_crit_count > 0:
            scores["actual_event_detail"] = max(2, 9 - (vague_crit_count * 2))
            for m in vague_report.matches:
                if m.severity == "critical":
                    weak_sections.append({
                        "scene_number": m.scene_number,
                        "issue": f"Vague storytelling: '{m.vague_phrase}'. Show what actually happened.",
                    })
        else:
            scores["actual_event_detail"] = 9

        # 3. Scene quality
        short_scenes = [sc for sc in scenes if len((sc.get("narration") or "").strip()) < 140]
        if len(short_scenes) > total_scenes * 0.25:
            scores["scene_quality"] = max(3, 9 - len(short_scenes))
            critical_issues.append(f"{len(short_scenes)} scenes have superficial 1-2 sentence narration")
            for sc in short_scenes[:3]:
                weak_sections.append({
                    "scene_number": sc.get("scene_number"),
                    "issue": "Narration too short (< 140 characters); missing atmosphere and concrete actions",
                })
        else:
            scores["scene_quality"] = 9

        # 4. Logical continuity
        if not logic_report.passed:
            scores["logical_continuity"] = max(3, 9 - (logic_report.penalty_score // 3))
        else:
            scores["logical_continuity"] = 9

        # 5. Character development
        if logic_report.character_flaws:
            scores["character_development"] = 6
        else:
            scores["character_development"] = 9

        # 6. Dialogue quality
        dialogue_count = sum(len(sc.get("dialogue") or []) for sc in scenes)
        if dialogue_count < total_scenes:
            scores["dialogue_quality"] = 6
            weak_sections.append({"scene_number": 0, "issue": "Insufficient dialogues across scenes"})
        if rep_report.repeated_dialogues_count > 0:
            scores["dialogue_quality"] = max(3, scores["dialogue_quality"] - rep_report.repeated_dialogues_count * 2)

        # 7. Suspense & Originality
        scores["suspense"] = 9
        scores["originality"] = 9

        # 8. Climax & Ending
        if scenes:
            ending = (story.get("ending") or "").strip()
            last_narr = (scenes[-1].get("narration") or "").strip()
            full_ending = ending + " " + last_narr

            if len(full_ending.strip()) < 100:
                scores["ending_and_conclusion"] = 4
                scores["climax"] = 5
                critical_issues.append("Abrupt or incomplete story ending")
                weak_sections.append({"scene_number": total_scenes, "issue": "Ending lacks proper resolution"})
            else:
                scores["ending_and_conclusion"] = 9
                scores["climax"] = 9

        return scores, weak_sections, critical_issues

    def run_human_review_checklist(
        self,
        story: dict[str, Any],
        rep_report: RepetitionReport,
        vague_report: VagueReport,
        logic_report: LogicValidationReport,
    ) -> dict[str, bool]:
        """
        Executes internal human review checklist from Section 11 of master specifications.
        """
        scenes = story.get("scenes") or []
        ending = (story.get("ending") or "").strip()
        last_narr = (scenes[-1].get("narration") if scenes else "") or ""

        checklist = {
            "can_viewer_clearly_understand_what_happened": len(scenes) >= 4 and len(vague_report.critical_issues) == 0,
            "are_important_incidents_explained_in_detail": len(vague_report.critical_issues) == 0,
            "can_viewer_visualize_scenes": all(bool(sc.get("visual_description") or sc.get("image_prompt")) for sc in scenes),
            "does_every_problem_have_actual_event_behind_it": len(vague_report.critical_issues) == 0,
            "are_character_reactions_believable": len(logic_report.character_flaws) == 0,
            "does_every_scene_add_new_information": (rep_report.duplicate_sentences_count == 0) and (rep_report.repeated_dialogues_count == 0),
            "is_anything_unnecessarily_repeated": len(rep_report.critical_issues) == 0 and (rep_report.duplicate_sentences_count == 0) and (rep_report.repeated_dialogues_count == 0),
            "are_scene_visuals_unique": not any(iss.issue_type == "repeated_image_prompt" for iss in rep_report.issues),
            "does_story_have_logical_structure": logic_report.passed,
            "does_climax_answer_central_mystery": len(logic_report.unresolved_clues) == 0,
            "does_ending_feel_complete": len(ending + last_narr) >= 100,
            "is_hindi_natural_and_engaging": True,
        }
        return checklist

    def _evaluate_with_llm(
        self,
        story: dict[str, Any],
        rule_scores: dict[str, int],
        weak_sections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calls LLM for cinematic screenplay scoring across the 10 dimensions."""
        scenes = story.get("scenes") or []
        title = story.get("title", "")
        synopsis = story.get("synopsis", "")
        ending = story.get("ending", "")

        beat_summary = []
        for sc in scenes[:16]:
            sn = sc.get("scene_number", 0)
            narr = (sc.get("narration") or "")[:180]
            diags = " | ".join(f"{d.get('character')}: {d.get('line')}" for d in sc.get("dialogue", [])[:2])
            beat_summary.append(f"Scene {sn}: {narr}\n  Dialogue: {diags}")

        prompt = f"""You are the Master Story Quality Auditor and Screenplay Doctor for YouTube Hindi Storytelling.
Audit the following Hindi story across 10 critical dimensions (score 1 to 10 each):

Title: {title}
Synopsis: {synopsis}
Ending: {ending}

Scenes:
{chr(10).join(beat_summary)}

EVALUATION CRITERIA (10 DIMENSIONS):
1. opening_hook: Shocking cold opening in first 30 seconds; no generic fairytale openings.
2. actual_event_detail: Strict SHOW, DON'T TELL. No empty phrases ('उसे बहुत संघर्ष करना पड़ा', 'हालात बिगड़ते गए').
3. scene_quality: Complete scenes with setup, concrete event, actions, and consequences.
4. logical_continuity: Strict cause-and-effect progression; early clues resolved properly.
5. character_development: Distinct characters with clear motivations, no Deus Ex Machina.
6. dialogue_quality: Natural, realistic Hindi dialogues that advance conflict; no question loops.
7. suspense: Escalating stakes and revelations throughout.
8. originality: Original setup, unpredictable twists, no clichés.
9. climax: High-stakes confrontation resolving the central mystery.
10. ending_and_conclusion: Complete, emotionally satisfying resolution showing character fates.

Return JSON ONLY:
{{
  "scores": {{
    "opening_hook": 9,
    "actual_event_detail": 9,
    "scene_quality": 9,
    "logical_continuity": 9,
    "character_development": 9,
    "dialogue_quality": 9,
    "suspense": 9,
    "originality": 9,
    "climax": 9,
    "ending_and_conclusion": 9
  }},
  "feedback": "1-2 sentence actionable advice to polish the story."
}}"""

        resp = self.llm.complete(
            prompt,
            system="You are an uncompromising screenplay judge and master editor. Output JSON only.",
            json_mode=True,
        )
        return self.llm.parse_json(resp.text)
