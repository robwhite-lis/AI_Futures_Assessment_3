"""
D&D SRD 5.2.1 Descriptive Text Extractor
==========================================
Extracts location/environment-descriptive passages from the D&D SRD 5.2.1.

The SRD is a technical rules document, not narrative prose. Its structure
requires different filters from the Morris extractor:

  Morris pipeline:  dialogue filter → attribution filter → descriptiveness score
  SRD pipeline:     mechanical density filter → stat block filter → descriptiveness score

The SRD's "descriptive" content is concentrated in:
  - Environmental Effects section (weather, terrain conditions)
  - Spell descriptions (terrain-altering and environment-creating spells)
  - Magic item descriptions (items that create or describe spaces)
  - Trap descriptions (which describe physical settings)
  - Adventure/GM guidance sections (terrain, travel, encounter building)

This difference is methodologically significant: the SRD represents the
rules framework the AI was likely trained on for generating fantasy locations.
Scoring AI-generated locations against SRD text tests whether the AI is
recapitulating its training vocabulary vs. developing novel description.

Author: Assessment 3 pipeline
"""

import re
import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict

try:
    import pypdf
except ImportError:
    print("ERROR: pypdf library not found. Please install it:")
    print("  pip install pypdf")
    exit(1)


# ---------------------------------------------------------------------------
# 0. CONFIGURABLE THRESHOLDS
# ---------------------------------------------------------------------------

MECHANICAL_RATIO_THRESHOLD = 0.25   # Paragraphs where >25% of words are mechanical
                                     # game terms → exclude
STAT_BLOCK_THRESHOLD       = 3      # Paragraphs with >= N stat-block markers → exclude
DICE_COUNT_THRESHOLD       = 2      # Paragraphs with >= N dice expressions → exclude
MIN_WORD_COUNT             = 25     # Ignore very short blocks (sub-headings, table rows)
MAX_WORD_COUNT             = 400    # Ignore very long blocks
DESCRIPTIVE_SCORE_MIN      = 0.15   # Lower than Morris: SRD prose is denser and less
                                     # adjective-rich, so we calibrate the threshold down

SPATIAL_VOCAB_BONUS        = 0.10
SPATIAL_BONUS_CAP          = 0.30
MECHANICAL_PENALTY         = 0.06   # Per mechanical term hit (capped)
MECHANICAL_PENALTY_CAP     = 0.24


# ---------------------------------------------------------------------------
# 1. LINGUISTIC RESOURCES
#    Adapted from Morris extractor with SRD-specific extensions.
# ---------------------------------------------------------------------------

# D&D game mechanics vocabulary — strong indicators of rules text rather
# than environment description. Presence penalises descriptiveness score.
MECHANICAL_VOCABULARY = {
    # Core game stats
    'hp', 'ac', 'dc', 'xp', 'cr',
    'strength', 'dexterity', 'constitution', 'intelligence', 'wisdom', 'charisma',
    'proficiency', 'initiative', 'multiattack',
    # Action economy
    'action', 'bonus', 'reaction', 'movement', 'speed',
    'cantrip', 'concentration', 'recharge',
    # Combat mechanics
    'saving', 'throw', 'attack', 'roll', 'damage', 'resistance', 'immunity',
    'advantage', 'disadvantage', 'modifier', 'exhaustion',
    # Conditions
    'blinded', 'charmed', 'deafened', 'frightened', 'grappled', 'incapacitated',
    'invisible', 'paralyzed', 'petrified', 'poisoned', 'prone', 'restrained',
    'stunned', 'unconscious',
    # Spell mechanics
    'casting', 'components', 'verbal', 'somatic', 'material', 'concentration',
    'duration', 'instantaneous', 'ritual', 'upcast',
    # Class/character mechanics
    'proficiency', 'bonus', 'feat', 'class', 'subclass', 'level', 'spell',
    'spellcasting', 'warlock', 'wizard', 'druid', 'cleric', 'bard', 'ranger',
    'fighter', 'rogue', 'monk', 'sorcerer', 'paladin', 'barbarian',
    # Creature categories
    'aberration', 'beast', 'celestial', 'construct', 'dragon', 'elemental',
    'fey', 'fiend', 'giant', 'humanoid', 'monstrosity', 'ooze', 'plant', 'undead',
}

# Adjective suffixes — same as Morris extractor
ADJECTIVE_SUFFIXES = (
    'ful', 'less', 'ous', 'ious', 'eous', 'ive', 'ative', 'itive',
    'al', 'ial', 'ical', 'ic', 'ish', 'en', 'ent', 'ant',
    'ary', 'ory', 'able', 'ible', 'ward', 'some',
)

# Descriptive adjectives — expanded to include D&D/fantasy register
DESCRIPTIVE_ADJECTIVES = {
    # General atmosphere
    'dark', 'bright', 'dim', 'shadowy', 'gloomy', 'luminous', 'radiant',
    'murky', 'misty', 'foggy', 'hazy', 'clear', 'opaque',
    # Temperature and weather
    'cold', 'frigid', 'icy', 'frozen', 'hot', 'warm', 'scorching', 'cool',
    'damp', 'wet', 'dry', 'humid', 'arid',
    # Terrain qualities
    'steep', 'rocky', 'sandy', 'muddy', 'slippery', 'rough', 'smooth',
    'flat', 'narrow', 'wide', 'deep', 'shallow', 'dense', 'thick', 'open',
    'enclosed', 'cavernous', 'vast', 'cramped', 'ancient', 'crumbling',
    # Magical/fantastical
    'magical', 'arcane', 'ethereal', 'shadowy', 'spectral', 'otherworldly',
    'supernatural', 'planar', 'divine', 'fiendish', 'eldritch', 'radiant',
    'enchanted', 'cursed', 'haunted', 'sacred', 'profane', 'hollow',
    # Physical properties
    'solid', 'soft', 'hard', 'brittle', 'heavy', 'light', 'translucent',
    'opaque', 'transparent', 'impenetrable', 'porous', 'dense',
    # Sensory
    'silent', 'loud', 'pungent', 'foul', 'sweet', 'fresh', 'stale', 'acrid',
    'putrid', 'fetid', 'earthy', 'damp',
    # Colours
    'black', 'white', 'grey', 'gray', 'red', 'blue', 'green', 'gold',
    'silver', 'purple', 'brown', 'crimson', 'azure', 'obsidian',
    # Size and scale
    'huge', 'massive', 'enormous', 'giant', 'towering', 'immense', 'vast',
    'tiny', 'small', 'large', 'great',
    # Condition
    'ruined', 'abandoned', 'desolate', 'barren', 'lush', 'verdant',
    'overgrown', 'pristine', 'crumbling', 'decayed', 'weathered',
    'fortified', 'imposing', 'forbidding', 'welcoming',
}

# Spatial/environmental vocabulary — SRD-specific additions to Morris list
SPATIAL_VOCABULARY = {
    # Natural terrain
    'forest', 'wood', 'woodland', 'trees', 'thicket', 'jungle', 'canopy',
    'hill', 'hills', 'mountain', 'mountains', 'peak', 'valley', 'ravine',
    'plain', 'plains', 'meadow', 'field', 'grassland', 'prairie',
    'moor', 'swamp', 'marsh', 'bog', 'fen', 'wetland',
    'desert', 'dune', 'wasteland', 'tundra', 'arctic', 'glacier', 'ice',
    'river', 'stream', 'waterfall', 'lake', 'sea', 'ocean', 'shore', 'coast',
    'cliff', 'canyon', 'gorge', 'chasm', 'crevasse',
    'cave', 'cavern', 'tunnel', 'passage', 'corridor', 'chamber', 'vault',
    'underground', 'underdark', 'subterranean',
    # Built environment
    'dungeon', 'tower', 'castle', 'fortress', 'citadel', 'stronghold',
    'ruin', 'temple', 'shrine', 'cathedral', 'crypt', 'tomb', 'mausoleum',
    'hall', 'chamber', 'room', 'cell', 'corridor', 'passage', 'stairway',
    'bridge', 'gate', 'door', 'portal', 'arch', 'wall', 'floor', 'ceiling',
    'column', 'pillar', 'altar', 'dais', 'throne', 'prison',
    'village', 'town', 'city', 'settlement', 'outpost', 'keep',
    # Planar/fantastical
    'plane', 'realm', 'domain', 'dimension', 'demiplane', 'void',
    'abyss', 'heavens', 'hell', 'feywild', 'shadowfell', 'ethereal',
    'astral', 'elemental',
    # Atmosphere
    'sky', 'air', 'darkness', 'shadow', 'light', 'fire', 'flame',
    'smoke', 'mist', 'fog', 'storm', 'wind', 'rain', 'snow', 'ice',
    'ground', 'earth', 'stone', 'rock', 'water', 'surface',
}

# Stat block markers — patterns strongly indicating monster/rules stat blocks
STAT_BLOCK_PATTERNS = [
    r'\b(STR|DEX|CON|INT|WIS|CHA)\b',
    r'\bHP\b.*\bSpeed\b',
    r'\bAC\b.*\bInitiative\b',
    r'\bCR\s+\d',
    r'\bPB\s+\+\d',
    r'Languages\b.*\bPassive Perception',
    r'\bMultiattack\b',
    r'\bMelee Attack Roll\b',
    r'\bRanged Attack Roll\b',
]


# ---------------------------------------------------------------------------
# 2. DATA STRUCTURE
# ---------------------------------------------------------------------------

@dataclass
class Passage:
    section: str          # Section/chapter heading context
    subsection: str       # Immediate subsection heading
    source_type: str      # 'environment', 'spell', 'magic_item', 'trap', 'general'
    text: str
    word_count: int
    mechanical_ratio: float
    dice_count: int
    descriptive_score: float
    spatial_hits: int
    mechanical_hits: int
    retained: bool
    exclusion_reason: str


# ---------------------------------------------------------------------------
# 3. TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """
    Repair SRD-specific text artifacts from PDF-to-text conversion.

    Key issues:
    1. Soft hyphen (U+0002 / \u0002) used as line-break hyphen in PDF extraction.
       These appear mid-word and must be removed to restore the original word.
       e.g. 'Consti\u0002tution' → 'Constitution'
    2. Page number headers: '123 System Reference Document 5.2.1'
    3. Hyphenated line breaks that the extractor didn't convert: legitimate
       hyphens at line end should be joined, soft hyphens removed.
    """
    # Remove soft hyphens (PDF line-break artifact)
    text = raw.replace('\u0002', '')

    # Remove page number headers
    text = re.sub(r'\n\d+ System Reference Document 5\.2\.1\n', '\n', text)

    # Join hyphenated line breaks: word-\nword → wordword
    # (these were genuine hyphenation breaks in the PDF layout)
    text = re.sub(r'-\n(\w)', r'\1', text)

    # Normalise multiple blank lines to double newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def strip_boilerplate(raw: str) -> str:
    """Remove legal header and table of contents."""
    # The SRD starts with legal info, then a TOC.
    # The TOC ends where the first actual content section begins.
    # TOC entries have the pattern: 'Title.....N'
    # First content section is 'Playing the Game' around line 508.
    # We find the start of the actual body by locating 'Playing the Game'
    # as a standalone section header (followed by a blank line and prose).

    # Strip legal block at top — ends before TOC
    # Find first occurrence of 'Playing the Game' as a section header
    # (not the TOC entry which has dot-leaders)
    body_start = re.search(r'\nPlaying the Game\n(?!\s*\.)', raw)
    if body_start:
        raw = raw[body_start.start():]

    return raw


# ---------------------------------------------------------------------------
# 4. STRUCTURE PARSING
# ---------------------------------------------------------------------------

def parse_into_chunks(body: str,
                      chunk_words: int = 80,
                      step_words: int  = 60) -> list[tuple[str, str, str]]:
    """
    Parse the SRD body into (section, subsection, chunk_text) tuples.

    The SRD was extracted from a two-column PDF layout, which means all lines
    are 30–60 character fragments of sentences — there are no reliable paragraph
    breaks. Standard paragraph-splitting fails entirely on this document.

    Strategy:
    1. Track section/subsection context by detecting genuine heading lines.
    2. Join all non-heading lines into a continuous text stream per subsection.
    3. Reconstruct the text into sentence-boundary chunks of ~chunk_words words,
       stepping forward by step_words to create mild overlap.
       Overlap ensures descriptions that straddle arbitrary cut points are
       represented in at least one chunk.

    Heading detection heuristic:
       A line is treated as a heading if it is:
       - 3–50 characters
       - Starts with a capital letter
       - Contains no dice notation (1d6 etc.)
       - Contains no mid-sentence punctuation suggesting it is a sentence fragment
         (no comma, no 'and'/'or' mid-line, no parenthesised game terms)
       - Does not end with a comma, conjunction, or preposition
    """
    # Source-type inference by subsection keywords
    SPELL_KEYWORDS = {
        'cloud', 'fog', 'darkness', 'wall', 'storm', 'terrain', 'mirage',
        'arcane', 'mansion', 'demiplane', 'conjure', 'awaken', 'plant',
        'spike', 'growth', 'entangle', 'pass', 'earth', 'control',
        'water', 'wind', 'ice', 'fire', 'thunder', 'lightning',
        'sanctuary', 'refuge',
    }
    MAGIC_ITEM_KEYWORDS = {
        'bag', 'cloak', 'hat', 'helm', 'boots', 'ring', 'staff',
        'wand', 'rod', 'cube', 'sphere', 'amulet', 'carpet',
    }
    ENVIRONMENT_KEYWORDS = {
        'cold', 'heat', 'altitude', 'precipitation', 'wind', 'ice',
        'water', 'terrain', 'hazard', 'travel', 'environment',
    }

    def infer_source_type(section: str, subsection: str) -> str:
        sl = (section + ' ' + subsection).lower()
        if 'spell' in sl:
            return 'spell'
        if 'magic item' in sl:
            return 'magic_item'
        if any(k in sl for k in ENVIRONMENT_KEYWORDS):
            return 'environment'
        if any(k in sl for k in SPELL_KEYWORDS):
            return 'spell'
        if any(k in sl for k in MAGIC_ITEM_KEYWORDS):
            return 'magic_item'
        return 'general'

    def is_heading(line: str, prev_line: str) -> bool:
        """Heuristic: is this line a standalone section/subsection heading?"""
        s = line.strip()
        if not s or len(s) < 3 or len(s) > 55:
            return False
        if not re.match(r'^[A-Z]', s):
            return False
        if re.search(r'\d+d\d+', s):          # dice notation
            return False
        if re.search(r'[.]{2,}', s):           # TOC dots
            return False
        if re.search(r'^\d+\s', s):            # starts with page number
            return False
        if s.endswith(',') or s.endswith(' and') or s.endswith(' or'):
            return False
        # Sentence fragments usually have a lowercase word after the first
        words = s.split()
        if len(words) > 10:                    # Too long to be a heading
            return False
        # If previous line ends mid-sentence (no period), this is likely a continuation
        ps = prev_line.strip()
        if ps and not ps.endswith('.') and not ps.endswith(':') and not ps.endswith('?'):
            if not re.match(r'^[A-Z]', s[1:2] if len(s) > 1 else ''):
                pass  # Allow headings even after mid-sentence lines
        return True

    def sentences_from_text(text: str) -> list[str]:
        """
        Split a block of joined lines into sentence-like units.
        Splits on '. ' or '.\n' followed by a capital letter,
        or on ':' introducing a list.
        """
        # Normalise whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Split on sentence boundaries
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Filter to non-empty
        return [p.strip() for p in parts if p.strip() and len(p.split()) > 3]

    def make_chunks(sentences: list[str],
                    chunk_words: int,
                    step_words: int) -> list[str]:
        """
        Group sentences into chunks of ~chunk_words words,
        stepping by ~step_words to create overlap.
        """
        if not sentences:
            return []
        chunks = []
        i = 0
        while i < len(sentences):
            chunk_sents = []
            wc = 0
            j = i
            while j < len(sentences):
                sw = len(sentences[j].split())
                if wc + sw > chunk_words and chunk_sents:
                    break
                chunk_sents.append(sentences[j])
                wc += sw
                j += 1
            if chunk_sents:
                chunks.append(' '.join(chunk_sents))
            # Step forward by step_words
            step_taken = 0
            while i < len(sentences) and step_taken < step_words:
                step_taken += len(sentences[i].split())
                i += 1
            if i == j:   # Safety: always advance at least one sentence
                i += 1
        return chunks

    # --- Main parsing loop ---
    lines = body.split('\n')
    results = []

    current_section    = 'Playing the Game'
    current_subsection = ''
    accumulated_lines  = []

    KNOWN_TOP_SECTIONS = {
        'Playing the Game', 'Equipment', 'Spells', 'Rules Glossary',
        'Gameplay Toolbox', 'Magic Items', 'Monsters', 'Running a Monster',
    }

    def flush(section, subsection):
        if not accumulated_lines:
            return
        joined = ' '.join(accumulated_lines)
        sents  = sentences_from_text(joined)
        chunks = make_chunks(sents, chunk_words, step_words)
        src    = infer_source_type(section, subsection)
        for c in chunks:
            results.append((section, subsection, src, c))

    prev_line = ''
    for raw_line in lines:
        line    = raw_line.strip()

        if not line:
            prev_line = line
            continue

        # Top-level section?
        if line in KNOWN_TOP_SECTIONS:
            flush(current_section, current_subsection)
            accumulated_lines  = []
            current_section    = line
            current_subsection = ''
            prev_line = line
            continue

        # Heading?
        if is_heading(line, prev_line):
            flush(current_section, current_subsection)
            accumulated_lines  = []
            current_subsection = line
            prev_line = line
            continue

        accumulated_lines.append(line)
        prev_line = line

    flush(current_section, current_subsection)
    return results


# ---------------------------------------------------------------------------
# 5. FILTERING FUNCTIONS
# ---------------------------------------------------------------------------

def has_stat_block(text: str) -> bool:
    """Detect if a paragraph is or contains a monster stat block."""
    for pattern in STAT_BLOCK_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def count_dice_expressions(text: str) -> int:
    """Count dice notation expressions (1d6, 2d10, 3d6+5, etc.)."""
    return len(re.findall(r'\b\d+d\d+', text))


def mechanical_ratio(text: str) -> tuple[float, int]:
    """
    Calculate proportion of words that are game mechanics vocabulary.
    Also counts mechanical term hits for scoring.
    Returns (ratio, hit_count)
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0, 0
    hits = sum(1 for w in words if w in MECHANICAL_VOCABULARY)
    return hits / len(words), hits


def score_descriptiveness(text: str) -> tuple[float, int, int]:
    """
    Score descriptiveness on 0-1 scale.
    Same method as Morris extractor, adapted thresholds.
    Returns (score, spatial_hits, mechanical_hits)
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
            adj_count += 0.5

    base_score = adj_count / len(words)

    spatial_hits   = sum(1 for w in words if w in SPATIAL_VOCABULARY)
    mechanical_hits = sum(1 for w in words if w in MECHANICAL_VOCABULARY)

    spatial_bonus      = min(spatial_hits * SPATIAL_VOCAB_BONUS, SPATIAL_BONUS_CAP)
    mechanical_penalty = min(mechanical_hits * MECHANICAL_PENALTY, MECHANICAL_PENALTY_CAP)

    score = base_score + spatial_bonus - mechanical_penalty
    return max(0.0, min(1.0, score)), spatial_hits, mechanical_hits


# ---------------------------------------------------------------------------
# 6. PDF TEXT EXTRACTION
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file using pypdf.
    Returns the concatenated text from all pages.
    """
    print(f"  Extracting text from PDF...")
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"  PDF has {num_pages} pages")

            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                if page_num % 50 == 0:  # Progress indicator
                    print(f"    Processing page {page_num}/{num_pages}...")
                text_parts.append(page.extract_text())

            full_text = '\n'.join(text_parts)
            return full_text
    except Exception as e:
        print(f"ERROR: Failed to extract text from PDF: {e}")
        exit(1)


# ---------------------------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str,
                 output_txt: str) -> None:

    print(f"Reading: {input_path}")

    # Check if input is PDF or text file
    if input_path.lower().endswith('.pdf'):
        raw = extract_text_from_pdf(input_path)
    else:
        raw = Path(input_path).read_text(encoding='utf-8', errors='replace')

    print(f"  Raw file size: {len(raw):,} chars")

    # Stage 1 — clean text artifacts
    cleaned = clean_text(raw)
    print(f"  After cleaning: {len(cleaned):,} chars")

    # Stage 2 — strip boilerplate (legal header + TOC)
    body = strip_boilerplate(cleaned)
    print(f"  After boilerplate strip: {len(body):,} chars")

    # Stage 3 — parse into sentence-window chunks with section metadata
    raw_chunks = parse_into_chunks(body, chunk_words=80, step_words=60)
    print(f"  Raw chunks generated: {len(raw_chunks)}")

    # Build candidate Passage objects, applying word-count filter
    candidates = []
    for section, subsection, source_type, text in raw_chunks:
        words = text.split()
        if MIN_WORD_COUNT <= len(words) <= MAX_WORD_COUNT:
            candidates.append(Passage(
                section=section,
                subsection=subsection,
                source_type=source_type,
                text=text,
                word_count=len(words),
                mechanical_ratio=0.0,
                dice_count=0,
                descriptive_score=0.0,
                spatial_hits=0,
                mechanical_hits=0,
                retained=False,
                exclusion_reason='',
            ))

    print(f"  Candidate paragraphs (word-count range "
          f"[{MIN_WORD_COUNT}–{MAX_WORD_COUNT}]): {len(candidates)}")

    # Stage 4 — apply filters
    for p in candidates:

        # Stat block filter — hard exclude
        if has_stat_block(p.text):
            p.exclusion_reason = 'stat_block_detected'
            continue

        # Dice expression filter — hard exclude if too mechanical
        p.dice_count = count_dice_expressions(p.text)
        if p.dice_count >= DICE_COUNT_THRESHOLD:
            p.exclusion_reason = (f'dice_count={p.dice_count} '
                                   f'(threshold={DICE_COUNT_THRESHOLD})')
            continue

        # Mechanical vocabulary ratio
        mech_ratio, mech_hits = mechanical_ratio(p.text)
        p.mechanical_ratio = mech_ratio
        p.mechanical_hits  = mech_hits
        if mech_ratio > MECHANICAL_RATIO_THRESHOLD:
            p.exclusion_reason = (f'mechanical_ratio={mech_ratio:.2f} '
                                   f'(threshold={MECHANICAL_RATIO_THRESHOLD})')
            continue

        # Descriptiveness score
        score, spatial_hits, mhits = score_descriptiveness(p.text)
        p.descriptive_score = score
        p.spatial_hits      = spatial_hits
        p.mechanical_hits   = mhits

        if score < DESCRIPTIVE_SCORE_MIN:
            p.exclusion_reason = (f'descriptive_score={score:.3f} '
                                   f'(threshold={DESCRIPTIVE_SCORE_MIN})')
            continue

        p.retained = True

    # Report
    retained = [p for p in candidates if p.retained]
    excluded = [p for p in candidates if not p.retained]

    print(f"\n  --- Results ---")
    print(f"  Total candidates: {len(candidates)}")
    print(f"  Retained:         {len(retained)}")
    print(f"  Excluded:         {len(excluded)}")

    reasons = {}
    for p in excluded:
        key = p.exclusion_reason.split('=')[0] if p.exclusion_reason else 'size_filter'
        reasons[key] = reasons.get(key, 0) + 1
    print(f"\n  Exclusion breakdown:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<35} {count}")

    if retained:
        scores = [p.descriptive_score for p in retained]
        print(f"\n  Score distribution:")
        print(f"    Min:  {min(scores):.3f}")
        print(f"    Max:  {max(scores):.3f}")
        print(f"    Mean: {sum(scores)/len(scores):.3f}")

    type_counts = {}
    for p in retained:
        type_counts[p.source_type] = type_counts.get(p.source_type, 0) + 1
    print(f"\n  Retained by source type:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<20} {count}")

    section_counts = {}
    for p in retained:
        section_counts[p.section] = section_counts.get(p.section, 0) + 1
    print(f"\n  Retained by section:")
    for s, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"    {s:<35} {count}")

    # Stage 5 — write outputs


    print(f"  Writing TXT -> {output_txt}")
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"D&D SRD 5.2.1 DESCRIPTIVE PASSAGES\n")
        f.write(f"Source: System Reference Document 5.2.1 (CC-BY-4.0)\n")
        f.write(f"Passages retained: {len(retained)}\n")
        f.write(f"Thresholds: mechanical<{MECHANICAL_RATIO_THRESHOLD}, "
                f"dice<{DICE_COUNT_THRESHOLD}, "
                f"descriptive>{DESCRIPTIVE_SCORE_MIN}\n")
        f.write("=" * 70 + "\n\n")

        for i, p in enumerate(retained, 1):
            f.write(f"[{i}] {p.section} | {p.subsection} "
                    f"[{p.source_type}]\n")
            f.write(f"    Score: {p.descriptive_score:.3f} | "
                    f"Words: {p.word_count} | "
                    f"Spatial: {p.spatial_hits} | "
                    f"Mechanical hits: {p.mechanical_hits} | "
                    f"Dice: {p.dice_count}\n")
            f.write(f"\n{p.text}\n")
            f.write("\n" + "-" * 70 + "\n\n")

    print(f"\nDone.")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline(
        input_path  = "./DD_SRD_CC_v5.pdf",
        output_txt  = "./srd_descriptive.txt",
    )
