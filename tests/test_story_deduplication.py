import pytest
from app.ai.story_generator import StoryGenerator


def test_story_generator_deduplicates_repetitive_dialogues():
    gen = StoryGenerator()
    story = {
        "title": "परीक्षण कहानी",
        "synopsis": "एक रहस्यमयी घटना",
        "scenes": [
            {
                "scene_number": 1,
                "narration": "रोहन ने पुरानी हवेली का भारी लकड़ी का दरवाज़ा धीरे से धक्का देकर खोला।",
                "dialogue": [{"character": "रोहन", "line": "यह क्या है?", "gender": "male"}],
                "mood": "suspense",
            },
            {
                "scene_number": 2,
                "narration": "अंदर धूल से भरी मेज़ पर एक लाल डायरी रखी हुई थी।",
                "dialogue": [{"character": "रोहन", "line": "यह क्या है?", "gender": "male"}],
                "mood": "suspense",
            },
            {
                "scene_number": 3,
                "narration": "अचानक पीछे से किसी के भारी कदमों की आहट सुनाई दी।",
                "dialogue": [{"character": "रोहन", "line": "यह क्या हो रहा है?", "gender": "male"}],
                "mood": "suspense",
            },
            {
                "scene_number": 4,
                "narration": "अंधेरे से एक अजनबी चेहरा निकलकर सामने आया।",
                "dialogue": [{"character": "रोहन", "line": "यह क्या हो रहा है?", "gender": "male"}],
                "mood": "suspense",
            },
        ],
    }

    cleaned = gen._clean_and_deduplicate(story)
    lines = [d["line"] for sc in cleaned["scenes"] for d in sc.get("dialogue", [])]

    # All dialogue lines must now be unique!
    assert len(lines) == len(set(lines)), f"Found duplicate dialogues: {lines}"
    # The first 'यह क्या है?' is preserved, but subsequent duplicates are replaced
    assert lines[0] == "यह क्या है?"
    assert lines[1] != "यह क्या है?"
    assert lines[2] == "यह क्या हो रहा है?"
    assert lines[3] != "यह क्या हो रहा है?"


def test_story_generator_enriches_short_narration_stubs():
    gen = StoryGenerator()
    story = {
        "title": "परीक्षण कहानी",
        "scenes": [
            {
                "scene_number": 1,
                "narration": "रोहन ने चाबी उठाई।",  # Very short stub (18 chars)
                "dialogue": [],
                "mood": "suspense",
            }
        ],
    }
    cleaned = gen._clean_and_deduplicate(story)
    narr = cleaned["scenes"][0]["narration"]
    assert len(narr) >= 100, f"Narration stub was not enriched: {narr}"
    assert "रोहन ने चाबी उठाई।" in narr


def test_calculate_spoken_chars():
    gen = StoryGenerator()
    story = {
        "scenes": [
            {
                "scene_number": 1,
                "narration": "एक दो तीन चार पाँच",
                "dialogue": [{"line": "छह सात आठ"}],
            }
        ]
    }
    chars = gen._calculate_spoken_chars(story)
    assert chars == len("एक दो तीन चार पाँच") + len("छह सात आठ")


def test_story_generator_enriches_multiple_stubs_uniquely():
    gen = StoryGenerator()
    story = {
        "title": "परीक्षण कहानी",
        "scenes": [
            {"scene_number": 1, "narration": "रोहन ने दरवाज़ा खोला।", "dialogue": []},
            {"scene_number": 2, "narration": "उसने अंदर कदम रखा।", "dialogue": []},
        ],
    }
    cleaned = gen._clean_and_deduplicate(story)
    s1_narr = cleaned["scenes"][0]["narration"]
    s2_narr = cleaned["scenes"][1]["narration"]
    assert s1_narr != s2_narr
    # The appended parts must not be identical
    assert s1_narr[25:] != s2_narr[25:]


def test_story_generator_deduplicates_identical_prompts_and_sentences():
    gen = StoryGenerator()
    story = {
        "title": "परीक्षण कहानी",
        "scenes": [
            {
                "scene_number": 1,
                "narration": "हवेली के अंदर गहरा सन्नाटा था। उसने सीढ़ियों पर कदम रखे।",
                "visual_description": "हवेली की सीढ़ियां",
                "image_prompt": "Cinematic dark haveli staircase",
                "dialogue": [],
            },
            {
                "scene_number": 2,
                "narration": "उसने सीढ़ियों पर कदम रखे। कमरे में एक पुराना संदूक रखा था।",
                "visual_description": "हवेली की सीढ़ियां",
                "image_prompt": "Cinematic dark haveli staircase",
                "dialogue": [],
            },
        ],
    }
    cleaned = gen._clean_and_deduplicate(story)
    # Sentence deduplication: scene 2 should not have "उसने सीढ़ियों पर कदम रखे।" repeated
    s2_narr = cleaned["scenes"][1]["narration"]
    assert s2_narr.count("उसने सीढ़ियों पर कदम रखे") == 0 or cleaned["scenes"][0]["narration"].count("उसने सीढ़ियों पर कदम रखे") == 1
    # Image prompt & visual description deduplication
    assert cleaned["scenes"][0]["image_prompt"] != cleaned["scenes"][1]["image_prompt"]
    assert cleaned["scenes"][0]["visual_description"] != cleaned["scenes"][1]["visual_description"]

