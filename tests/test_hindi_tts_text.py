from app.voice.hindi_text import prepare_hindi_tts, split_hindi_for_tts


def test_comma_splits_without_dropping_words():
    text = "रोहित की माँ ने कहा, यहाँ आओ।"
    prep = prepare_hindi_tts(text)
    joined = " ".join(prep.segments)
    assert "रोहित" in joined
    assert "माँ" in joined
    assert "यहाँ" in joined
    assert len(prep.segments) >= 2


def test_two_sentences():
    text = "रोहित की माँ ने कहा, यहाँ आओ। रोहित ने पूछा, क्या हुआ माँ?"
    segs = split_hindi_for_tts(text)
    assert len(segs) >= 3
    assert any("रोहित की" in s for s in segs)


def test_dialogue_manager_narration_clean():
    import re
    from app.voice.dialogue_manager import DialogueManager
    # Verify DialogueManager can be imported without NameError on re
    dm = DialogueManager()
    assert dm is not None


def test_dialogue_attribution_speech_retained():
    from app.ai.story_generator import StoryGenerator

    # Case 1: Narrative speech clause should keep 'राघव ने कहा' in spoken text
    char, emo, line = StoryGenerator.clean_dialogue_line("राघव ने कहा: कौन है वहाँ?")
    assert char == "राघव"
    assert "राघव ने कहा" in line
    assert "कौन है वहाँ?" in line

    # Case 2: 'उसने कहा' should also be retained in spoken text
    char, emo, line = StoryGenerator.clean_dialogue_line("उसने कहा: मुझे जाने दो।")
    assert "उसने कहा" in line
    assert "मुझे जाने दो" in line

    # Case 3: Script tag format without narrative verbs strips colon and keeps clean dialogue
    char, emo, line = StoryGenerator.clean_dialogue_line("राघव: कौन है वहाँ?")
    assert char == "राघव"
    assert line == "कौन है वहाँ?"
    assert ":" not in line

