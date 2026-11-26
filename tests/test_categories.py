from app.content.categories import CATEGORIES, get_category, format_duration_range
from app.content.category_config import load_category_config


def test_all_categories_have_weights():
    cfg = load_category_config()
    for cid in cfg.get("enabled_categories", []):
        assert cid in CATEGORIES


def test_format_duration_long():
    lo, hi, min_s, max_s = format_duration_range("long")
    assert 500 <= lo <= 700
    assert min_s < max_s


def test_get_category_default():
    c = get_category("unknown_id")
    assert c.id == "haunted_haveli"


def test_story_generator_has_master_horror_rules():
    from app.ai.story_generator import StoryGenerator
    from app.content.categories import category_story_block

    gen = StoryGenerator()
    sys_prompt = gen._hindi_system()
    assert "MASTER SYSTEM PROMPT: HORROR & MYSTERY STORY GENERATOR" in sys_prompt
    assert "परछाई किसकी थी?" in sys_prompt
    assert "ALLOWED STORY CATEGORIES" in sys_prompt
    assert "CINEMATIC STORY STRUCTURE" in sys_prompt

    block = category_story_block(get_category("haunted_haveli"))
    assert "HIGH RETENTION HORROR & SUSPENSE DIRECTIVES" in block
