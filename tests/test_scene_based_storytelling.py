from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.ai.repetition_detector import RepetitionDetector
from app.ai.vague_sentence_detector import VagueSentenceDetector
from app.ai.story_logic_validator import StoryLogicValidator
from app.ai.story_blueprint_generator import StoryBlueprintGenerator
from app.ai.story_quality_auditor import StoryQualityAuditor
from app.ai.story_rewriter import StoryRewriter
from app.ai.story_generator import StoryGenerator, StoryGenerationRequest


def test_repetition_detector_catches_duplicate_sentences_and_paragraphs():
    detector = RepetitionDetector()
    story = {
        "title": "दोहराव वाली कहानी",
        "scenes": [
            {
                "scene_number": 1,
                "narration": "राघव ने पुरानी हवेली का भारी लकड़ी का दरवाज़ा धीरे से धक्का देकर खोला। अंदर गहरा अंधेरा फैला हुआ था।",
                "dialogue": [{"character": "राघव", "line": "दरवाज़ा खुला है।"}],
            },
            {
                "scene_number": 2,
                "narration": "राघव ने पुरानी हवेली का भारी लकड़ी का दरवाज़ा धीरे से धक्का देकर खोला। अंदर धूल भरी मेज़ रखी थी।",
                "dialogue": [{"character": "राघव", "line": "दरवाज़ा खुला है।"}],
            },
        ],
    }

    report = detector.check_story(story)
    assert report.has_repetition is True
    assert report.duplicate_sentences_count >= 1
    assert report.repeated_dialogues_count >= 1
    assert report.penalty_score > 0
    assert any(i.issue_type == "duplicate_sentence" for i in report.issues)


def test_repetition_detector_catches_semantic_concept_loops():
    detector = RepetitionDetector()
    story = {
        "title": "अर्थगत दोहराव",
        "scenes": [
            {
                "scene_number": 1,
                "narration": "उस रात के बाद राघव की जिंदगी बदल गई। चारों तरफ गहरा सन्नाटा पसरा हुआ था।",
                "dialogue": [],
            },
            {
                "scene_number": 2,
                "narration": "अब सब कुछ बदल चुका था। उसके जीवन में बड़ा बदलाव आ गया।",
                "dialogue": [],
            },
            {
                "scene_number": 3,
                "narration": "अब उसकी दुनिया पहले जैसी नहीं रही। सब कुछ बदल गया।",
                "dialogue": [],
            },
        ],
    }

    report = detector.check_story(story)
    assert report.semantic_repeats_count >= 2
    assert any(i.issue_type == "semantic_repetition" for i in report.issues)


def test_vague_sentence_detector_flags_empty_storytelling():
    detector = VagueSentenceDetector()
    vague_story = {
        "title": "कमजोर कहानी",
        "scenes": [
            {
                "scene_number": 1,
                "narration": "उसे जिंदगी में बहुत मुश्किलों का सामना करना पड़ा। उसकी परेशानियां बढ़ती चली गईं। हालात बहुत खराब हो गए। उसे कई समस्याओं से गुजरना पड़ा।",
                "dialogue": [],
            }
        ],
        "ending": "और इस तरह उसकी जिंदगी बदल गई।",
    }

    report = detector.check_story(vague_story)
    assert report.has_vague_storytelling is True
    assert report.vague_count >= 2
    assert report.penalty_score > 0
    assert len(report.critical_issues) >= 1
    assert any("मुश्किलों का सामना" in m.vague_phrase or "हालात बहुत खराब" in m.vague_phrase for m in report.matches)


def test_vague_sentence_detector_passes_concrete_scene():
    detector = VagueSentenceDetector()
    concrete_story = {
        "title": "ठोस कहानी",
        "scenes": [
            {
                "scene_number": 1,
                "narration": "राहुल उस दिन ऑफिस से लौटा तो घर के बाहर उसके चाचा कैलाश खड़े थे। उनके हाथ में कुछ कागज़ात और पुश्तैनी मकान की चाबी थी।",
                "dialogue": [
                    {"character": "राहुल", "line": "चाचा, आप यहाँ इस वक्त क्यों खड़े हैं?"},
                    {"character": "कैलाश", "line": "यह मकान अब तुम्हारे नाम नहीं रहेगा। तुम्हारे पिता ने इसे मेरे पास गिरवी रखा था।"}
                ],
            }
        ],
        "ending": "राहुल ने थाने पहुँचकर इंस्पेक्टर शर्मा को वसीयत की मूल प्रति और वीडियो रिकॉर्डिंग सौंपी, जिसके बाद कैलाश को गिरफ्तार कर लिया गया।",
    }

    report = detector.check_story(concrete_story)
    assert len(report.critical_issues) == 0
    assert report.penalty_score == 0


def test_story_logic_validator_flags_unintroduced_character_at_climax():
    validator = StoryLogicValidator()
    story = {
        "title": "तार्किक दोष",
        "characters": [
            {"name": "राघव", "gender": "male"}
        ],
        "scenes": [
            {"scene_number": 1, "narration": "राघव कमरे में दाखिल हुआ।", "dialogue": [{"character": "राघव", "line": "यहाँ कौन है?"}]},
            {"scene_number": 2, "narration": "राघव ने खत देखा।", "dialogue": [{"character": "राघव", "line": "खत में क्या लिखा है?"}]},
            {"scene_number": 3, "narration": "राघव ने तिजोरी खोली।", "dialogue": [{"character": "राघव", "line": "तिजोरी खाली है।"}]},
            {"scene_number": 4, "narration": "राघव सीढ़ियों पर चढ़ा।", "dialogue": [{"character": "राघव", "line": "ऊपर आवाज़ आ रही है।"}]},
            {"scene_number": 5, "narration": "अचानक विक्रम ने आकर सब कुछ सुलझा दिया।", "dialogue": [{"character": "विक्रम", "line": "मैंने ही तुम्हें बचाया है।"}]},
        ],
        "ending": "सब ठीक हो गया।",
    }

    report = validator.validate(story)
    assert report.passed is False
    assert any("विक्रम" in issue for issue in report.critical_issues)
    assert any("सब ठीक हो गया" in issue for issue in report.critical_issues)


def test_story_blueprint_generator_produces_full_narrative_architecture():
    gen = StoryBlueprintGenerator()
    bp = gen.generate_procedural_blueprint("हवेली का रहस्य", "भूतिया हवेली")
    bp_dict = bp.to_dict()

    assert "concept" in bp_dict
    assert "main_character" in bp_dict
    assert "supporting_characters" in bp_dict
    assert "main_conflict" in bp_dict
    assert "antagonist_or_opposing_force" in bp_dict
    assert len(bp_dict["important_events"]) >= 3
    assert len(bp_dict["hidden_clues"]) >= 2
    assert "major_twist" in bp_dict
    assert "climax" in bp_dict
    assert "final_resolution" in bp_dict
    assert "conclusion_moral" in bp_dict


def test_story_rewriter_fixes_vague_phrases():
    rewriter = StoryRewriter()
    auditor = StoryQualityAuditor()

    vague_story = {
        "title": "परीक्षण कहानी",
        "scenes": [
            {
                "scene_number": 1,
                "narration": "राघव को बहुत संघर्ष करना पड़ा। उसके साथ बहुत बुरा हुआ।",
                "dialogue": [{"character": "राघव", "line": "यह क्या है?"}],
            }
        ],
        "ending": "समय के साथ सब ठीक हो गया।",
    }

    report = auditor.audit(vague_story)
    improved_story = rewriter.rewrite_and_improve(vague_story, report)

    s1_narr = improved_story["scenes"][0]["narration"]
    assert "बहुत संघर्ष करना पड़ा" not in s1_narr
    assert "समय के साथ सब ठीक हो गया" not in improved_story["ending"]
    assert len(s1_narr) > 100


def test_story_quality_auditor_and_human_review():
    auditor = StoryQualityAuditor()
    gen = StoryGenerator()
    req = StoryGenerationRequest(language="hi", genre="horror")
    fallback_story = gen._generate_fallback_story(req, {"main_topic": "हवेली का बंद कमरा: परछाई किसकी थी?"})

    report = auditor.audit(fallback_story)
    assert report.total_score >= 85
    assert report.passed is True
    assert report.repetition_penalty == 0
    assert report.vague_penalty == 0
    assert len(report.critical_issues) == 0

    # Test Section 11 Human Review checklist
    assert report.human_review_checklist.get("can_viewer_clearly_understand_what_happened") is True
    assert report.human_review_checklist.get("are_important_incidents_explained_in_detail") is True
    assert report.human_review_checklist.get("does_climax_answer_central_mystery") is True
    assert report.human_review_checklist.get("does_ending_feel_complete") is True


def test_story_quality_auditor_fails_repetitive_and_vague_story():
    auditor = StoryQualityAuditor()
    bad_story = {
        "title": "खराब कहानी",
        "scenes": [
            {
                "scene_number": 1,
                "narration": "उसे जिंदगी में बहुत मुश्किलों का सामना करना पड़ा। उसकी परेशानियां बढ़ती चली गईं।",
                "dialogue": [{"character": "नायक", "line": "यह क्या है?"}],
            },
            {
                "scene_number": 2,
                "narration": "उसे जिंदगी में बहुत मुश्किलों का सामना करना पड़ा। उसकी परेशानियां बढ़ती चली गईं।",
                "dialogue": [{"character": "नायक", "line": "यह क्या है?"}],
            },
        ],
        "ending": "यही जिंदगी की सच्चाई है।",
    }

    report = auditor.audit(bad_story)
    assert report.passed is False
    assert report.total_score < 85
    assert report.repetition_penalty > 0
    assert report.vague_penalty > 0
    assert len(report.critical_issues) > 0
