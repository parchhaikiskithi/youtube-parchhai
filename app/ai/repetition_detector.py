from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RepetitionIssue:
    issue_type: str  # "duplicate_sentence", "repeated_paragraph", "semantic_repetition", "repeated_dialogue", "repetitive_intro"
    scene_number: int
    text: str
    details: str
    severity: str = "warning"  # "warning" or "critical"


@dataclass
class RepetitionReport:
    has_repetition: bool
    penalty_score: int
    issues: list[RepetitionIssue] = field(default_factory=list)
    critical_issues: list[str] = field(default_factory=list)
    duplicate_sentences_count: int = 0
    repeated_dialogues_count: int = 0
    semantic_repeats_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_repetition": self.has_repetition,
            "penalty_score": self.penalty_score,
            "issues": [
                {
                    "issue_type": i.issue_type,
                    "scene_number": i.scene_number,
                    "text": i.text,
                    "details": i.details,
                    "severity": i.severity,
                }
                for i in self.issues
            ],
            "critical_issues": self.critical_issues,
            "duplicate_sentences_count": self.duplicate_sentences_count,
            "repeated_dialogues_count": self.repeated_dialogues_count,
            "semantic_repeats_count": self.semantic_repeats_count,
        }


class RepetitionDetector:
    """
    Detects lexical, syntactic, and semantic repetition across Hindi story scenes.
    Flags duplicate sentences, repeated paragraphs, semantic concept loops,
    and repetitive character dialogues.
    """

    # Semantic concept clusters of phrases that convey the exact same meaning without advancing the plot
    SEMANTIC_CLUSTERS = {
        "life_changed": [
            r"जिंदगी बदल गई",
            r"सब कुछ बदल चुका था",
            r"अब उसकी दुनिया पहले जैसी नहीं रही",
            r"जीवन में बड़ा बदलाव आ गया",
            r"सब कुछ बदल गया",
            r"जिंदगी पूरी तरह बदल चुकी थी",
            r"जीवन हमेशा के लिए बदल गया",
            r"दुनिया बदल चुकी थी",
        ],
        "mounting_troubles": [
            r"मुश्किलों का सामना करना पड़ा",
            r"परेशानियां बढ़ती चली गईं",
            r"हालात बहुत खराब हो गए",
            r"समस्याओं से गुजरना पड़ा",
            r"मुसीबतों का पहाड़ टूट पड़ा",
            r"हालात बिगड़ते चले गए",
            r"एक के बाद एक मुसीबतें",
            r"कठिनाइयों का सामना किया",
            r"परेशानियों का दौर",
            r"मुसीबतें खत्म होने का नाम नहीं ले रही थीं",
        ],
        "vague_suffering": [
            r"उसने बहुत कुछ सहा",
            r"बहुत संघर्ष करना पड़ा",
            r"उसके साथ बहुत बुरा हुआ",
            r"जिंदगी में तूफान आ गया",
            r"दर्द सहना पड़ा",
            r"दुखों का सामना",
        ],
        "physical_fear_reaction": [
            r"रोंगटे खड़े हो गए",
            r"हाथ कांपने लगे",
            r"पसीना छूटने लगा",
            r"दिल दहल उठा",
            r"सांसें अटक गईं",
            r"पैरों तले ज़मीन खिसक गई",
            r"होश उड़ गए",
            r"गले में आवाज़ अटक गई",
            r"खून सूख गया",
        ],
        "eerie_atmosphere": [
            r"गहरा सन्नाटा पसरा हुआ था",
            r"माहौल में अजीब बेचैनी थी",
            r"खामोशी रूह कंपा रही थी",
            r"चारों तरफ सन्नाटा था",
            r"सन्नाटा चीख रहा था",
            r"अजीब सी दहशत फैली हुई थी",
        ],
        "unanswered_questions": [
            r"सवालों का कोई जवाब नहीं था",
            r"मन में कई सवाल उठ रहे थे",
            r"उलझनें बढ़ती जा रही थीं",
            r"कुछ समझ नहीं आ रहा था",
            r"सवालों का बवंडर",
        ],
        "generic_questions": [
            r"यह क्या है",
            r"यह क्या हो रहा है",
            r"तुम कौन हो",
            r"यह कौन है",
            r"यह क्या मतलब है",
            r"अब क्या होगा",
            r"मुझे नहीं पता",
            r"देखते हैं क्या होता है",
        ],
    }

    GENERIC_DIALOGUE_PATTERNS = [
        re.compile(r"^(?:यह|ये)\s+क्या\s+है[?।!\s]*$"),
        re.compile(r"^(?:यह|ये)\s+क्या\s+हो\s+रहा\s+है[?।!\s]*$"),
        re.compile(r"^तुम\s+कौन\s+हो[?।!\s]*$"),
        re.compile(r"^वहाँ\s+कौन\s+है[?।!\s]*$"),
        re.compile(r"^कौन\s+है[?।!\s]*$"),
        re.compile(r"^क्या\s+हुआ[?।!\s]*$"),
        re.compile(r"^अब\s+क्या\s+होगा[?।!\s]*$"),
        re.compile(r"^मुझे\s+नहीं\s+पता[?।!\s]*$"),
        re.compile(r"^मुझे\s+कुछ\s+समझ\s+नहीं\s+आ\s+रहा[?।!\s]*$"),
    ]

    SITUATION_GROUNDED_DIALOGUES = [
        "दरवाज़े की कुंडी पर खून के ताज़ा धब्बे हैं... कोई अभी-अभी यहाँ से भागा है।",
        "इस खत पर पिताजी की मुहर लगी है, लेकिन तारीख कल रात की है!",
        "तिजोरी खुली है और वसीयत के पन्ने फर्श पर बिखरे पड़े हैं।",
        "तुमने मुझसे यह सच क्यों छुपाया कि हवेली का पिछला हिस्सा तीस साल से सील था?",
        "गाड़ी के पहियों के निशान जंगल की पुरानी बावड़ी की तरफ जा रहे हैं।",
        "आईने के पीछे एक गुप्त ताला लगा हुआ है, इसकी चाबी किसके पास है?",
        "अगर पुलिस ने यह कागज़ात देख लिए, तो हममें से कोई भी बेकसूर साबित नहीं हो पाएगा।",
        "सीढ़ियों के नीचे एक लाल बक्सा दबा हुआ है... उसमें किसी की पुरानी डायरी है।",
    ]

    @staticmethod
    def normalize_hindi_text(text: str) -> str:
        """Strips punctuation, trailing spaces, and normalizes Hindi orthographic variations."""
        t = re.sub(r"[।?!.,:;\-\"'`‘’“”\(\)\[\]\{\}\/\\_]", " ", text or "")
        t = re.sub(r"\s+", " ", t).strip()
        # Common vowel/nasalization normalizations
        t = t.replace("हैं", "है").replace("हूँ", "हूं").replace("यहाँ", "यहां").replace("वहाँ", "वहां")
        return t

    @classmethod
    def split_into_sentences(cls, text: str) -> list[str]:
        """Splits Hindi prose into sentences using Devanagari danda and standard punctuation."""
        raw_parts = re.split(r"[।?!;\n]+", text or "")
        sentences = [p.strip() for p in raw_parts if len(p.strip()) > 8]
        return sentences

    def check_story(self, story: dict[str, Any]) -> RepetitionReport:
        """
        Analyzes full story JSON for all repetition vectors:
        - Duplicate sentences across scenes
        - Repeated paragraphs (Jaccard similarity > 0.65)
        - Semantic repetition clusters (same idea repeated across multiple scenes)
        - Repeated dialogue lines
        - Repeated scene openers
        """
        scenes = story.get("scenes") or []
        issues: list[RepetitionIssue] = []
        critical_issues: list[str] = []

        seen_sentences: dict[str, int] = {}
        seen_dialogues: dict[str, int] = {}
        cluster_occurrences: dict[str, list[tuple[int, str]]] = {c: [] for c in self.SEMANTIC_CLUSTERS}

        duplicate_sentences_count = 0
        repeated_dialogues_count = 0
        semantic_repeats_count = 0

        # 1. Analyze Narrations for Sentence Duplication & Semantic Loops
        scene_paragraphs: list[tuple[int, str, list[str]]] = []

        for sc in scenes:
            sn = sc.get("scene_number", 0)
            narr = (sc.get("narration") or "").strip()
            if not narr:
                continue

            sentences = self.split_into_sentences(narr)
            scene_paragraphs.append((sn, narr, sentences))

            for sent in sentences:
                norm = self.normalize_hindi_text(sent)
                if len(norm) < 15:
                    continue

                if norm in seen_sentences:
                    first_sn = seen_sentences[norm]
                    duplicate_sentences_count += 1
                    issues.append(
                        RepetitionIssue(
                            issue_type="duplicate_sentence",
                            scene_number=sn,
                            text=sent,
                            details=f"Exact/near-duplicate of sentence in Scene {first_sn}",
                            severity="critical",
                        )
                    )
                else:
                    seen_sentences[norm] = sn

                # Check semantic clusters
                for cluster_name, patterns in self.SEMANTIC_CLUSTERS.items():
                    for pat in patterns:
                        if re.search(pat, sent):
                            cluster_occurrences[cluster_name].append((sn, sent))
                            break

        # 2. Check Repeated Paragraphs (High Jaccard Word Similarity)
        for i in range(len(scene_paragraphs)):
            sn_i, narr_i, _ = scene_paragraphs[i]
            words_i = set(self.normalize_hindi_text(narr_i).split())
            if len(words_i) < 10:
                continue
            for j in range(i + 1, len(scene_paragraphs)):
                sn_j, narr_j, _ = scene_paragraphs[j]
                words_j = set(self.normalize_hindi_text(narr_j).split())
                if len(words_j) < 10:
                    continue
                intersection = words_i.intersection(words_j)
                union = words_i.union(words_j)
                similarity = len(intersection) / len(union) if union else 0.0
                if similarity >= 0.60:
                    issues.append(
                        RepetitionIssue(
                            issue_type="repeated_paragraph",
                            scene_number=sn_j,
                            text=narr_j[:120] + "...",
                            details=f"Paragraph has {int(similarity * 100)}% vocabulary overlap with Scene {sn_i}",
                            severity="critical",
                        )
                    )
                    critical_issues.append(f"Scene {sn_j} repeats narration from Scene {sn_i} ({int(similarity*100)}% overlap)")

        # 3. Check Semantic Clusters (Same idea repeated > 1 time without new development)
        for cluster_name, matches in cluster_occurrences.items():
            if len(matches) > 1:
                # If repeated across multiple scenes, flag as semantic repetition
                semantic_repeats_count += len(matches) - 1
                scenes_involved = sorted(list({m[0] for m in matches}))
                if len(scenes_involved) > 1:
                    first_sn, first_txt = matches[0]
                    for sn_m, txt_m in matches[1:]:
                        issues.append(
                            RepetitionIssue(
                                issue_type="semantic_repetition",
                                scene_number=sn_m,
                                text=txt_m,
                                details=f"Repeats '{cluster_name}' sentiment already expressed in Scene {first_sn}",
                                severity="critical" if len(matches) >= 3 else "warning",
                            )
                        )
                    if len(matches) >= 3:
                        critical_issues.append(
                            f"Semantic concept '{cluster_name}' repeated {len(matches)} times across scenes {scenes_involved}"
                        )

        # 4. Check Dialogue Repetition
        for sc in scenes:
            sn = sc.get("scene_number", 0)
            dialogues = sc.get("dialogue") or []
            for d in dialogues:
                if not isinstance(d, dict):
                    continue
                raw_line = (d.get("line") or "").strip()
                if not raw_line:
                    continue
                norm_dial = self.normalize_hindi_text(raw_line)
                if not norm_dial:
                    continue

                # Check generic question patterns
                for pat in self.GENERIC_DIALOGUE_PATTERNS:
                    if pat.match(raw_line.strip()) or pat.match(norm_dial):
                        issues.append(
                            RepetitionIssue(
                                issue_type="generic_dialogue",
                                scene_number=sn,
                                text=raw_line,
                                details="Generic low-effort dialogue pattern (e.g., 'यह क्या है?', 'तुम कौन हो?')",
                                severity="warning",
                            )
                        )

                # Check duplicate lines across scenes
                if norm_dial in seen_dialogues:
                    first_sn = seen_dialogues[norm_dial]
                    repeated_dialogues_count += 1
                    issues.append(
                        RepetitionIssue(
                            issue_type="repeated_dialogue",
                            scene_number=sn,
                            text=raw_line,
                            details=f"Dialogue is identical to line in Scene {first_sn}",
                            severity="critical",
                        )
                    )
                else:
                    seen_dialogues[norm_dial] = sn

        # 5. Check Scene Openings
        scene_openers: dict[str, list[int]] = {}
        for sc in scenes:
            sn = sc.get("scene_number", 0)
            narr = (sc.get("narration") or "").strip()
            if narr:
                words = narr.split()
                if len(words) >= 4:
                    opener = " ".join(words[:4])
                    scene_openers.setdefault(opener, []).append(sn)

        for opener, sns in scene_openers.items():
            if len(sns) > 2:
                issues.append(
                    RepetitionIssue(
                        issue_type="repetitive_intro",
                        scene_number=sns[1],
                        text=opener,
                        details=f"Formulaic scene opener '{opener}' repeated in scenes {sns}",
                        severity="warning",
                    )
                )

        # 6. Check Duplicate Image Prompts & Visual Descriptions
        seen_prompts: dict[str, int] = {}
        repeated_prompts_count = 0
        for sc in scenes:
            sn = sc.get("scene_number", 0)
            ip = (sc.get("image_prompt") or "").strip().lower()
            if ip and len(ip) > 25:
                if ip in seen_prompts:
                    first_sn = seen_prompts[ip]
                    repeated_prompts_count += 1
                    issues.append(
                        RepetitionIssue(
                            issue_type="repeated_image_prompt",
                            scene_number=sn,
                            text=sc.get("image_prompt", "")[:120],
                            details=f"Image prompt is identical to Scene {first_sn}",
                            severity="critical",
                        )
                    )
                else:
                    seen_prompts[ip] = sn

        if duplicate_sentences_count > 0:
            critical_issues.append(f"Found {duplicate_sentences_count} duplicate sentence(s) across scenes")
        if repeated_dialogues_count > 0:
            critical_issues.append(f"Found {repeated_dialogues_count} repeated dialogue line(s) across scenes")
        if repeated_prompts_count > 0:
            critical_issues.append(f"Found {repeated_prompts_count} repeated image prompt(s) across scenes")

        # 7. Check Repeated 3-4 word phrases (Crutch Phrases)
        phrase_tracker: dict[str, list[int]] = {}
        for sc in scenes:
            sn = sc.get("scene_number", 0)
            narr = (sc.get("narration") or "").strip()
            if narr:
                for sent in self.split_into_sentences(narr):
                    words = [w for w in re.findall(r"[\u0900-\u097F]+", sent) if len(w) > 1]
                    for n in (3, 4):
                        for idx in range(len(words) - n + 1):
                            p = " ".join(words[idx : idx + n])
                            if any(k in p for k in ("होता जा रहा", "होती जा रही", "कर दिया था", "जा रहा था", "गया था", "और उसने अपने", "और उसने", "ने अपने", "उसने अपने")):
                                continue
                            phrase_tracker.setdefault(p, []).append(sn)

        repeated_phrases_count = 0
        for p, sns in phrase_tracker.items():
            unique_scenes = set(sns)
            if len(unique_scenes) >= 3:
                repeated_phrases_count += 1
                issues.append(
                    RepetitionIssue(
                        issue_type="repeated_phrase",
                        scene_number=sns[1],
                        text=p,
                        details=f"Phrase '{p}' repeated across {len(unique_scenes)} scenes: {sorted(list(unique_scenes))}",
                        severity="critical" if len(unique_scenes) >= 4 else "warning",
                    )
                )
                if len(unique_scenes) >= 4:
                    critical_issues.append(f"Crutch phrase '{p}' repeated across {len(unique_scenes)} scenes")

        # 8. Check Excessive Word Repetition within a Single Scene
        stopwords = {
            "और", "था", "थी", "थे", "है", "हैं", "के", "की", "का", "को", "में", "से", "पर", "ने", "एक",
            "कि", "भी", "तो", "ही", "हो", "कर", "रहा", "रही", "रहे", "गया", "गई", "गए", "लिया", "दिया",
            "हुए", "हुआ", "हुई", "जो", "वह", "उसके", "उसकी", "उसके", "उसने", "उसे", "अपने", "अपनी", "अपना",
            "यहाँ", "वहाँ", "अब", "जब", "तब", "कुछ", "कोई", "क्या", "कौन", "तक"
        }
        for sc in scenes:
            sn = sc.get("scene_number", 0)
            narr = (sc.get("narration") or "").strip()
            if narr:
                words = [w for w in re.findall(r"[\u0900-\u097F]+", narr) if len(w) >= 3]
                content_words = [w for w in words if w not in stopwords]
                word_counts = {}
                for w in content_words:
                    word_counts[w] = word_counts.get(w, 0) + 1
                for w, cnt in word_counts.items():
                    if any(c.get("name") == w for c in story.get("characters", [])):
                        continue
                    if cnt >= 3:
                        issues.append(
                            RepetitionIssue(
                                issue_type="repetitive_word",
                                scene_number=sn,
                                text=w,
                                details=f"Word '{w}' repeated {cnt} times within Scene {sn} narration",
                                severity="critical" if cnt >= 4 else "warning",
                            )
                        )

        # Calculate penalty score
        penalty = 0
        penalty += duplicate_sentences_count * 5
        penalty += repeated_dialogues_count * 4
        penalty += repeated_prompts_count * 5
        penalty += semantic_repeats_count * 2
        penalty += repeated_phrases_count * 2
        for iss in issues:
            if iss.issue_type == "repeated_paragraph":
                penalty += 6
            elif iss.issue_type == "generic_dialogue":
                penalty += 1
            elif iss.issue_type == "repetitive_word" and iss.severity == "critical":
                penalty += 3

        penalty = min(penalty, 30)
        has_rep = len(issues) > 0

        report = RepetitionReport(
            has_repetition=has_rep,
            penalty_score=penalty,
            issues=issues,
            critical_issues=critical_issues,
            duplicate_sentences_count=duplicate_sentences_count,
            repeated_dialogues_count=repeated_dialogues_count,
            semantic_repeats_count=semantic_repeats_count,
        )
        return report

    def deduplicate_story(self, story: dict[str, Any]) -> dict[str, Any]:
        """
        Cleans and fixes duplicate sentences, repeated dialogues, and formulaic questions in-place.
        """
        scenes = story.get("scenes") or []
        seen_sentences: set[str] = set()
        seen_dialogues: set[str] = set()
        pool_idx = 0

        for sc in scenes:
            sn = sc.get("scene_number", 0)
            narr = (sc.get("narration") or "").strip()
            if narr:
                sentences = self.split_into_sentences(narr)
                unique_sentences = []
                for sent in sentences:
                    norm = self.normalize_hindi_text(sent)
                    if len(norm) >= 15:
                        if norm in seen_sentences:
                            logger.info("Scene %s: Removing duplicate sentence '%s'", sn, sent)
                            continue
                        seen_sentences.add(norm)
                    unique_sentences.append(sent)

                if unique_sentences:
                    sc["narration"] = "। ".join(unique_sentences).strip()
                    if not sc["narration"].endswith("।"):
                        sc["narration"] += "।"

            # Clean dialogues
            dialogues = sc.get("dialogue") or []
            new_dialogue = []
            for d in dialogues:
                if not isinstance(d, dict):
                    continue
                raw_line = (d.get("line") or "").strip()
                if not raw_line:
                    continue
                norm = self.normalize_hindi_text(raw_line)

                # Replace only true duplicate dialogues seen earlier
                is_duplicate = norm in seen_dialogues

                if is_duplicate:
                    replacement = self.SITUATION_GROUNDED_DIALOGUES[pool_idx % len(self.SITUATION_GROUNDED_DIALOGUES)]
                    pool_idx += 1
                    logger.info("Scene %s: Replacing duplicate dialogue '%s' -> '%s'", sn, raw_line, replacement)
                    d["line"] = replacement
                    seen_dialogues.add(self.normalize_hindi_text(replacement))
                else:
                    seen_dialogues.add(norm)
                new_dialogue.append(d)
            sc["dialogue"] = new_dialogue

        # Clean duplicate image prompts and visual descriptions across scenes
        seen_prompts: set[str] = set()
        seen_visuals: set[str] = set()
        for sc in scenes:
            sn = sc.get("scene_number", 0)
            ip = (sc.get("image_prompt") or "").strip()
            vd = (sc.get("visual_description") or "").strip()
            narr_snippet = (sc.get("narration") or "")[:60].replace('"', "'").strip()

            if ip:
                norm_ip = ip.lower()
                if norm_ip in seen_prompts:
                    new_ip = f"{ip}, distinct Scene {sn} dramatic perspective, focused on: {narr_snippet or 'immediate action'}"
                    logger.info("Scene %s: Differentiating duplicate image prompt", sn)
                    sc["image_prompt"] = new_ip
                else:
                    seen_prompts.add(norm_ip)

            if vd:
                norm_vd = vd.lower()
                if norm_vd in seen_visuals:
                    new_vd = f"{vd} (सीन {sn}: नया कैमरा एंगल व विवरण)"
                    logger.info("Scene %s: Differentiating duplicate visual description", sn)
                    sc["visual_description"] = new_vd
                else:
                    seen_visuals.add(norm_vd)

        # Diversify repetitive crutch phrases across narrations
        crutch_phrases = {
            r"को लगता है कि": [
                "को आभास हुआ कि",
                "ने महसूस किया कि",
                "के मन में शंका उभरी कि",
                "को यकीन हो चला था कि",
            ],
            r"खड़ा होता है": [
                "ठिठक जाता है",
                "कदम रोक लेता है",
                "सन्नाटे में जम जाता है",
            ],
            r"देखता है कि": [
                "की नज़र पड़ती है कि",
                "ने गौर से देखा कि",
                "के सामने साफ़ था कि",
            ],
            r"भयानक सच्चाई का पता लगाता है": [
                "रोंगटे खड़े कर देने वाले सच से पर्दा उठाता है",
                "एक दिल दहला देने वाली हकीकत से रूबरू होता है",
                "उस खौफ़नाक रहस्य की तह तक पहुँचता है",
            ],
        }
        crutch_counters = {pat: 0 for pat in crutch_phrases}
        for sc in scenes:
            narr = sc.get("narration") or ""
            if narr:
                for pat, alts in crutch_phrases.items():
                    if re.search(pat, narr):
                        def make_sub(p, alt_list):
                            def _sub(m):
                                idx = crutch_counters[p]
                                crutch_counters[p] += 1
                                if idx == 0:
                                    return m.group(0)
                                return alt_list[(idx - 1) % len(alt_list)]
                            return _sub
                        narr = re.sub(pat, make_sub(pat, alts), narr)
                sc["narration"] = narr

        return story
