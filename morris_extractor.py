"""
Morris Descriptive Text Extractor
==================================
Extracts location-descriptive passages from 'The Well at the World's End'
by William Morris (Project Gutenberg edition).

Pipeline:
  1. Strip Gutenberg boilerplate and table of contents
  2. Segment into paragraphs with Book/Chapter metadata
  3. Filter out dialogue-heavy paragraphs
  4. Filter out action/narrative paragraphs
  5. Score remaining paragraphs for descriptiveness
  6. Output filtered corpus as JSON, CSV, and plain text

All heuristics are rule-based (no external model downloads required),
making every scoring decision transparent and auditable.

Author: Assessment 3 pipeline
"""

import re
import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# 0. CONFIGURABLE THRESHOLDS
#    Adjust these to tune the extraction. Values documented with rationale.
# ---------------------------------------------------------------------------

DIALOGUE_RATIO_THRESHOLD   = 0.30   # Paragraphs where >30% of chars are quoted speech → exclude
ATTRIBUTION_THRESHOLD      = 3      # Paragraphs with >= N dialogue attribution verbs → penalise
                                    # (relaxed from 2 → 3 after inspecting borderline cases:
                                    #  attribution=2 passages are often mixed narrative/description
                                    #  and still contribute useful descriptive content)
MIN_WORD_COUNT             = 30     # Ignore very short paragraphs (headings, fragments)
MAX_WORD_COUNT             = 500    # Ignore unusually long blocks (likely merged paragraphs)
DESCRIPTIVE_SCORE_MIN      = 0.18   # Minimum descriptiveness score to retain a paragraph
                                    # (calibrated by inspecting the score distribution:
                                    #  passages 0.18–0.25 were manually validated as genuinely
                                    #  descriptive; passages below 0.18 were predominantly
                                    #  narrative action or character interiority)
SPATIAL_VOCAB_BONUS        = 0.10   # Score boost per spatial vocabulary match (capped)
SPATIAL_BONUS_CAP          = 0.30   # Maximum total bonus from spatial vocabulary
ACTION_VERB_PENALTY        = 0.08   # Score reduction per action verb hit (capped)
ACTION_PENALTY_CAP         = 0.24   # Maximum total penalty from action verbs


# ---------------------------------------------------------------------------
# 1. LINGUISTIC RESOURCES
#    Curated word lists built for Morris's medieval/romance register.
#    References: standard adjective suffix patterns (Quirk et al., 1985,
#    'A Comprehensive Grammar of the English Language') + domain vocabulary.
# ---------------------------------------------------------------------------

# Adjective suffixes — standard morphological markers for English adjectives.
# Some produce false positives (e.g. '-al' in 'oval') but systematic error
# is preferable to biased error for this scoring purpose.
ADJECTIVE_SUFFIXES = (
    'ful', 'less', 'ous', 'ious', 'eous', 'ive', 'ative', 'itive',
    'al', 'ial', 'ical', 'ic', 'ish', 'en', 'ent', 'ant',
    'ary', 'ory', 'able', 'ible', 'ward', 'some',
)

# High-frequency descriptive adjectives in Morris's prose register.
# Drawn from sample passages — these appear in location/atmosphere descriptions.
DESCRIPTIVE_ADJECTIVES = {
    'fair', 'goodly', 'great', 'tall', 'dark', 'deep', 'wide', 'long',
    'short', 'high', 'low', 'broad', 'narrow', 'steep', 'flat', 'bare',
    'bright', 'dim', 'pale', 'black', 'white', 'grey', 'green', 'golden',
    'rich', 'poor', 'old', 'new', 'ancient', 'still', 'quiet', 'loud',
    'cold', 'warm', 'hot', 'cool', 'wet', 'dry', 'rough', 'smooth',
    'heavy', 'light', 'soft', 'hard', 'clear', 'murky', 'shadowy',
    'sweet', 'bitter', 'strong', 'weak', 'fresh', 'stale', 'noble',
    'goodsome', 'forlorn', 'desolate', 'pleasant', 'woeful', 'merry',
    'sad', 'glad', 'dreadful', 'terrible', 'wondrous', 'marvellous',
    'mighty', 'little', 'huge', 'vast', 'open', 'close', 'thick',
    'thin', 'round', 'square', 'straight', 'winding', 'broken', 'whole',
    'painted', 'gilded', 'hung', 'carved', 'wrought', 'built', 'paved',
    'stony', 'grassy', 'leafy', 'woody', 'sandy', 'rocky', 'mossy',
    'gloomy', 'cheerful', 'sunlit', 'shadowed', 'moonlit', 'windswept',
}

# Spatial and environmental vocabulary — the vocabulary domain of location
# description in medieval romance. Presence signals a location-descriptive passage.
SPATIAL_VOCABULARY = {
    # Landscape
    'forest', 'wood', 'woodland', 'trees', 'thicket', 'grove', 'copse',
    'hill', 'hills', 'mountain', 'mountains', 'valley', 'vale', 'plain',
    'plains', 'meadow', 'meadows', 'field', 'fields', 'moor', 'marsh',
    'river', 'stream', 'brook', 'water', 'lake', 'pool', 'cliff', 'shore',
    'beach', 'sand', 'rock', 'rocks', 'stone', 'stones', 'path', 'road',
    'track', 'way', 'bridge', 'ford', 'bank', 'slope', 'crest', 'ridge',
    'ness', 'headland', 'moorland', 'downland', 'upland', 'lowland',
    # Built environment
    'hall', 'house', 'castle', 'tower', 'gate', 'wall', 'walls', 'door',
    'chamber', 'room', 'court', 'courtyard', 'garden', 'garth', 'yard',
    'street', 'square', 'market', 'town', 'village', 'burg', 'city',
    'abbey', 'church', 'minster', 'chapel', 'inn', 'hostel', 'guesthouse',
    'stair', 'steps', 'arch', 'roof', 'floor', 'window', 'hearth', 'fire',
    'pillar', 'column', 'portal', 'threshold', 'parapet', 'battlement',
    # Atmosphere / light / time
    'sky', 'sun', 'moon', 'stars', 'light', 'shadow', 'darkness', 'dusk',
    'dawn', 'evening', 'morning', 'night', 'day', 'mist', 'fog', 'rain',
    'wind', 'storm', 'cloud', 'clouds', 'air', 'silence', 'sound',
    # Morris-specific spatial terms
    'garth', 'croft', 'mead', 'lade', 'mere', 'covert', 'shaw',
    'linn', 'fell', 'wold', 'weald', 'heath', 'brae', 'lea', 'leas',
    'toft', 'bourne', 'burn', 'knowe', 'scaur', 'ghyll',
}

# Dialogue attribution verbs — Morris's usage for introducing or closing speech.
ATTRIBUTION_VERBS = {
    'said', 'quoth', 'spake', 'spoke', 'answered', 'replied', 'cried',
    'called', 'asked', 'bade', 'told', 'laughed', 'sighed', 'whispered',
    'shouted', 'exclaimed', 'murmured', 'saying',
}

# Action and motion verbs — high frequency in narrative/action paragraphs.
# Their presence (especially as the main verb) signals non-descriptive prose.
ACTION_VERBS = {
    'rode', 'riding', 'ride', 'ridden',
    'came', 'come', 'coming',
    'went', 'go', 'going', 'gone',
    'led', 'lead', 'leading',
    'followed', 'follow', 'following',
    'departed', 'depart', 'departing',
    'entered', 'enter', 'entering',
    'left', 'leave', 'leaving',
    'turned', 'turn', 'turning',
    'passed', 'pass', 'passing',
    'crossed', 'cross', 'crossing',
    'reached', 'reach', 'reaching',
    'climbed', 'climb', 'climbing',
    'descended', 'descend', 'descending',
    'mounted', 'mount', 'mounting',
    'ran', 'run', 'running',
    'walked', 'walk', 'walking',
    'hastened', 'hasten', 'hastening',
    'slew', 'slay', 'slaying',
    'fought', 'fight', 'fighting',
    'fled', 'flee', 'fleeing',
    'brought', 'bring', 'bringing',
    'took', 'take', 'taking',
    'made', 'make', 'making',
    'gave', 'give', 'giving',
    'sent', 'send', 'sending',
    'saw', 'see', 'seeing',
    'heard', 'hear', 'hearing',
}


# ---------------------------------------------------------------------------
# 2. DATA STRUCTURE
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    book: str
    chapter_num: int
    chapter_title: str
    text: str
    word_count: int
    dialogue_ratio: float
    attribution_count: int
    descriptive_score: float
    spatial_hits: int
    action_hits: int
    retained: bool
    exclusion_reason: str


# ---------------------------------------------------------------------------
# 3. EXTRACTION FUNCTIONS
# ---------------------------------------------------------------------------

def strip_boilerplate(raw: str) -> str:
    """Remove Gutenberg header, footer, and table of contents."""
    start_marker = '*** START OF THE PROJECT GUTENBERG EBOOK'
    end_marker   = '*** END OF THE PROJECT GUTENBERG EBOOK'

    start = raw.find(start_marker)
    end   = raw.find(end_marker)

    if start == -1 or end == -1:
        raise ValueError("Could not locate Gutenberg markers in text.")

    body = raw[start:end]

    # The table of contents precedes the first actual BOOK/CHAPTER body text.
    # The TOC contains "BOOK ONE  The Road Unto Love" (single line).
    # The body contains "BOOK ONE\n\nThe Road Unto Love" (separated by newlines).
    # We find the first occurrence of BOOK ONE followed by a newline then blank line.
    toc_end = re.search(r'BOOK ONE\s*\n\s*\n', body)
    if toc_end:
        body = body[toc_end.start():]

    return body


def parse_structure(body: str) -> list[Paragraph]:
    """
    Walk the body text, tracking Book/Chapter context, and split
    content into paragraph-level Paragraph objects.
    """
    paragraphs = []
    current_book    = "BOOK ONE"
    current_ch_num  = 0
    current_ch_title = ""

    # Split on 2+ newlines to get blocks
    blocks = re.split(r'\n{2,}', body)

    for block in blocks:
        text = block.strip()
        if not text:
            continue

        # ---- Detect BOOK header ----
        book_match = re.match(r'^BOOK (ONE|TWO|THREE|FOUR|FIVE)\s*$', text)
        if book_match:
            current_book = f"BOOK {book_match.group(1)}"
            continue

        # ---- Detect CHAPTER header (number only) ----
        chapter_num_match = re.match(r'^CHAPTER (\d+)\s*$', text)
        if chapter_num_match:
            current_ch_num = int(chapter_num_match.group(1))
            continue

        # ---- Detect chapter title (short line after CHAPTER N, no punctuation mid-text) ----
        # Chapter titles are typically 3–12 words, title-cased, no quotes
        if (len(text.split()) <= 15
                and '\n' not in text
                and not text.startswith('"')
                and re.match(r'^[A-Z]', text)
                and current_ch_num > 0
                and text == current_ch_title or len(text) < 100):
            # Heuristic: if it's short and we just saw a chapter number, treat as title
            if len(text) < 120 and not any(c in text for c in [',', ';']) :
                # Only update if it looks like a title (not narrative prose)
                word_count_quick = len(text.split())
                if word_count_quick <= 12:
                    current_ch_title = text
                    continue

        # ---- Book subtitle lines (e.g. "The Road Unto Love") ----
        # These appear right after BOOK headers — short, no quotes, title-like
        if len(text.split()) <= 8 and '\n' not in text and re.match(r'^The |^Road', text):
            continue

        # ---- Actual paragraph content ----
        words = text.split()
        word_count = len(words)

        if word_count < MIN_WORD_COUNT or word_count > MAX_WORD_COUNT:
            continue

        para = Paragraph(
            book=current_book,
            chapter_num=current_ch_num,
            chapter_title=current_ch_title,
            text=text,
            word_count=word_count,
            dialogue_ratio=0.0,
            attribution_count=0,
            descriptive_score=0.0,
            spatial_hits=0,
            action_hits=0,
            retained=False,
            exclusion_reason="",
        )
        paragraphs.append(para)

    return paragraphs


def score_dialogue_ratio(text: str) -> float:
    """
    Calculate proportion of characters that fall within double-quoted speech.
    Morris uses straight double quotes consistently throughout the Gutenberg text.
    """
    quoted_chars = sum(len(m.group(0)) for m in re.finditer(r'"[^"]*"', text))
    return quoted_chars / len(text) if text else 0.0


def count_attribution_verbs(text: str) -> int:
    """Count occurrences of dialogue attribution verbs in the paragraph."""
    words_lower = re.findall(r'\b\w+\b', text.lower())
    return sum(1 for w in words_lower if w in ATTRIBUTION_VERBS)


def score_descriptiveness(text: str) -> tuple[float, int, int]:
    """
    Score how descriptive a paragraph is on a 0–1 scale.

    Method:
      base_score  = (adjective_suffix_matches + known_descriptive_adj) / word_count
      spatial_bonus = min(spatial_hits * SPATIAL_VOCAB_BONUS, SPATIAL_BONUS_CAP)
      action_penalty = min(action_hits * ACTION_VERB_PENALTY, ACTION_PENALTY_CAP)
      final = base_score + spatial_bonus - action_penalty  (clipped to [0, 1])

    Returns: (score, spatial_hits, action_hits)
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0, 0, 0

    adj_count = 0
    for word in words:
        if word in DESCRIPTIVE_ADJECTIVES:
            adj_count += 1
        elif any(word.endswith(suffix) and len(word) > len(suffix) + 2
                 for suffix in ADJECTIVE_SUFFIXES):
            adj_count += 0.5   # Partial credit for suffix match (less reliable)

    base_score = adj_count / len(words)

    spatial_hits = sum(1 for w in words if w in SPATIAL_VOCABULARY)
    action_hits  = sum(1 for w in words if w in ACTION_VERBS)

    spatial_bonus  = min(spatial_hits * SPATIAL_VOCAB_BONUS, SPATIAL_BONUS_CAP)
    action_penalty = min(action_hits * ACTION_VERB_PENALTY, ACTION_PENALTY_CAP)

    score = base_score + spatial_bonus - action_penalty
    return max(0.0, min(1.0, score)), spatial_hits, action_hits


# ---------------------------------------------------------------------------
# 4. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str,
                 output_txt: str) -> None:

    print(f"Reading: {input_path}")
    raw = Path(input_path).read_text(encoding='utf-8')

    # Stage 1 — strip boilerplate
    body = strip_boilerplate(raw)
    print(f"  Body length after stripping boilerplate: {len(body):,} chars")

    # Stage 2 — parse into paragraphs with metadata
    paragraphs = parse_structure(body)
    print(f"  Paragraphs in word-count range "
          f"[{MIN_WORD_COUNT}–{MAX_WORD_COUNT}]: {len(paragraphs)}")

    # Stages 3–5 — score each paragraph
    for para in paragraphs:

        # Dialogue ratio
        para.dialogue_ratio = score_dialogue_ratio(para.text)
        if para.dialogue_ratio > DIALOGUE_RATIO_THRESHOLD:
            para.exclusion_reason = (
                f"dialogue_ratio={para.dialogue_ratio:.2f} "
                f"(threshold={DIALOGUE_RATIO_THRESHOLD})"
            )
            continue

        # Attribution verb count
        para.attribution_count = count_attribution_verbs(para.text)
        if para.attribution_count >= ATTRIBUTION_THRESHOLD:
            para.exclusion_reason = (
                f"attribution_count={para.attribution_count} "
                f"(threshold={ATTRIBUTION_THRESHOLD})"
            )
            continue

        # Descriptiveness score
        score, spatial_hits, action_hits = score_descriptiveness(para.text)
        para.descriptive_score = score
        para.spatial_hits      = spatial_hits
        para.action_hits       = action_hits

        if score < DESCRIPTIVE_SCORE_MIN:
            para.exclusion_reason = (
                f"descriptive_score={score:.3f} "
                f"(threshold={DESCRIPTIVE_SCORE_MIN})"
            )
            continue

        para.retained = True

    # Report
    retained   = [p for p in paragraphs if p.retained]
    excluded   = [p for p in paragraphs if not p.retained]

    print(f"\n  --- Results ---")
    print(f"  Total scored:  {len(paragraphs)}")
    print(f"  Retained:      {len(retained)}")
    print(f"  Excluded:      {len(excluded)}")

    # Breakdown of exclusion reasons
    reasons = {}
    for p in excluded:
        key = p.exclusion_reason.split('=')[0] if p.exclusion_reason else 'size_filter'
        reasons[key] = reasons.get(key, 0) + 1
    print(f"\n  Exclusion breakdown:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<30} {count}")

    # Score distribution of retained passages
    if retained:
        scores = [p.descriptive_score for p in retained]
        print(f"\n  Retained score distribution:")
        print(f"    Min:  {min(scores):.3f}")
        print(f"    Max:  {max(scores):.3f}")
        print(f"    Mean: {sum(scores)/len(scores):.3f}")

    # Book distribution
    book_counts = {}
    for p in retained:
        book_counts[p.book] = book_counts.get(p.book, 0) + 1
    print(f"\n  Retained by book:")
    for book, count in sorted(book_counts.items()):
        print(f"    {book}: {count}")

    # Stage 6 — write outputs

    # Plain text — clean corpus for embedding
    print(f"  Writing TXT  → {output_txt}")
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"MORRIS DESCRIPTIVE PASSAGES\n")
        f.write(f"Extracted from: The Well at the World's End\n")
        f.write(f"Passages retained: {len(retained)}\n")
        f.write(f"Thresholds: dialogue<{DIALOGUE_RATIO_THRESHOLD}, "
                f"attribution<{ATTRIBUTION_THRESHOLD}, "
                f"descriptive>{DESCRIPTIVE_SCORE_MIN}\n")
        f.write("=" * 70 + "\n\n")

        for i, para in enumerate(retained, 1):
            f.write(f"[{i}] {para.book} | Ch.{para.chapter_num} "
                    f"— {para.chapter_title}\n")
            f.write(f"    Score: {para.descriptive_score:.3f} | "
                    f"Words: {para.word_count} | "
                    f"Spatial hits: {para.spatial_hits} | "
                    f"Action hits: {para.action_hits}\n")
            f.write(f"\n{para.text}\n")
            f.write("\n" + "-" * 70 + "\n\n")

    print(f"\nDone.")


# ---------------------------------------------------------------------------
# 5. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline(
        input_path  = "./The Well at the Worlds End.txt",
        output_txt  = "./morris_descriptive.txt",
    )
