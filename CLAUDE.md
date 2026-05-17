# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Futures Assessment 3 project that analyzes descriptive text from different sources using word embeddings and natural language processing. The project compares:

1. **William Morris's literary prose** ("The Well at the World's End") - medieval/romance fantasy writing
2. **D&D System Reference Document 5.2.1** - technical game rules with environmental descriptions
3. **AI-generated location descriptions** - synthetic fantasy location text

The goal is to extract descriptive passages from source texts and compare them using word2vec embeddings to determine whether AI-generated text resembles human literary writing or recapitulates training data vocabulary.

## Core Pipeline Scripts

### morris_extractor.py
Extracts location-descriptive passages from William Morris's "The Well at the World's End" (Project Gutenberg edition).

**Run the extractor:**
```bash
python morris_extractor.py
```

**Key features:**
- Strips Gutenberg boilerplate and table of contents
- Segments into paragraphs with Book/Chapter metadata
- Filters dialogue-heavy paragraphs (>30% quoted speech)
- Filters action/narrative paragraphs (uses action verb detection)
- Scores descriptiveness using adjective density + spatial vocabulary
- Outputs to: `morris_descriptive.txt`

**Thresholds (configurable at top of file):**
- `DIALOGUE_RATIO_THRESHOLD = 0.30` - Exclude paragraphs with >30% dialogue
- `ATTRIBUTION_THRESHOLD = 3` - Exclude paragraphs with ≥3 attribution verbs
- `DESCRIPTIVE_SCORE_MIN = 0.18` - Minimum descriptiveness score to retain
- `MIN_WORD_COUNT = 30` / `MAX_WORD_COUNT = 500` - Word count range

### srd_extractor.py
Extracts environment/location-descriptive passages from D&D SRD 5.2.1 PDF.

**Run the extractor:**
```bash
python srd_extractor.py
```

**Key differences from Morris extractor:**
- Uses mechanical vocabulary filter instead of dialogue filter
- Filters stat blocks and dice notation (game mechanics)
- Lower descriptiveness threshold (0.15 vs 0.18) - SRD prose is denser
- Chunks text using sliding window (80-word chunks, 60-word step) because PDF has no paragraph breaks
- Outputs to: `srd_descriptive.txt`

**Thresholds:**
- `MECHANICAL_RATIO_THRESHOLD = 0.25` - Exclude if >25% words are game mechanics terms
- `DICE_COUNT_THRESHOLD = 2` - Exclude if ≥2 dice expressions (1d6, 2d10, etc.)
- `DESCRIPTIVE_SCORE_MIN = 0.15` - Lower than Morris due to technical register

## Analysis Notebooks

### compare.ipynb
Main analysis notebook that compares different text corpora using word2vec embeddings.

**Key sections:**
1. **Setup** - Imports gensim, nltk, sklearn, plotly for NLP and visualization
2. **Text preprocessing** - Tokenization, lemmatization, stopword removal
3. **word2vec training** - Trains embeddings on source text
4. **Exploration** - Similarity queries, word relationships
5. **Visualization** - PCA dimension reduction + 3D scatter plots (plotly)
6. **Pre-trained models** - Uses Google News word2vec (300-dimensional, 3M words)

**Typical workflow:**
```python
# Load text
with open('morris_descriptive.txt', 'r') as f:
    text = f.read()

# Preprocess
text_tokens = [word_tokenize(sent) for sent in text.split('.')]
text_lemmas = [[lemmatizer.lemmatize(w) for w in sent
                if w not in stops and w not in punct]
               for sent in text_tokens]

# Train model
model = gensim.models.Word2Vec(text_lemmas, min_count=20, vector_size=300)

# Explore
model.wv.most_similar(['castle'])
model.wv.similarity('forest', 'woodland')
```

### word2vec.ipynb
Educational notebook demonstrating word2vec concepts using the Morris text. Similar structure to compare.ipynb but includes more pedagogical explanations.

## Data Files

### Source Texts
- `The Well at the Worlds End.txt` - Full Project Gutenberg text (William Morris)
- `DD_SRD_CC_v5.pdf` - D&D System Reference Document 5.2.1 (PDF format)

### Extracted Descriptive Text
- `morris_descriptive.txt` - Filtered descriptive passages from Morris (~47KB)
- `srd_descriptive.txt` - Filtered descriptive passages from SRD (~258KB)

### AI-Generated Data
- `locations.txt` - AI-generated fantasy location descriptions (name | description format)
- `structures.txt` - AI-generated fantasy structure descriptions (name | description format)

These appear to be synthetic training data or test corpora for comparison with human-written text.

## Development Workflow

### Running Extractors
1. Place source texts in root directory
2. Run extractor scripts: `python morris_extractor.py` or `python srd_extractor.py`
3. Check console output for statistics (paragraphs retained/excluded, score distributions)
4. Output files written to root directory

### Working with Notebooks
The notebooks use standard data science stack:
- **gensim** for word2vec models
- **nltk** for text preprocessing (tokenization, lemmatization, stopwords)
- **sklearn** for PCA, clustering (KMeans, AgglomerativeClustering)
- **plotly** for interactive 3D visualizations
- **pandas/numpy** for data manipulation

### Methodological Notes

**Why two different extractors?**
The Morris and SRD texts require fundamentally different filtering approaches:
- Morris: Narrative prose → filter dialogue/action → retain descriptive passages
- SRD: Technical rules document → filter mechanics/stat blocks → retain environmental descriptions

**Scoring philosophy:**
All heuristics are rule-based and transparent (no ML-based classification). This ensures every filtering decision is auditable and reproducible.

**Threshold calibration:**
Thresholds were set by manual inspection of score distributions. Comments in code explain the rationale for each threshold value.

## Architecture Notes

### Text Processing Pipeline (Extractors)
```
Raw text
  → Strip boilerplate (headers, TOC, legal text)
  → Segment into units (paragraphs for Morris, sliding windows for SRD)
  → Apply hard filters (dialogue/mechanics, stat blocks)
  → Score descriptiveness (adjective density + spatial vocab + penalties)
  → Retain passages above threshold
  → Write output (plain text with metadata headers)
```

### Descriptiveness Scoring Algorithm
```python
base_score = (adjective_count) / word_count
spatial_bonus = min(spatial_hits * BONUS, BONUS_CAP)
penalty = min(action_hits * PENALTY, PENALTY_CAP)  # Morris uses action verbs
# OR
penalty = min(mechanical_hits * PENALTY, PENALTY_CAP)  # SRD uses mechanical terms
final_score = clip(base_score + spatial_bonus - penalty, 0, 1)
```

### Linguistic Resources
Both extractors maintain curated word lists:
- `DESCRIPTIVE_ADJECTIVES` - High-frequency descriptive adjectives in each corpus
- `SPATIAL_VOCABULARY` - Environmental/location vocabulary (terrain, structures, atmosphere)
- `ADJECTIVE_SUFFIXES` - Morphological patterns for adjective detection
- `ACTION_VERBS` (Morris) or `MECHANICAL_VOCABULARY` (SRD) - Terms that indicate non-descriptive text

These lists are domain-specific and were curated for each text's register (medieval romance vs. D&D fantasy).

## Git Repository Structure

```
.
├── morris_extractor.py          # Morris text extractor
├── srd_extractor.py             # SRD extractor
├── compare.ipynb                # Main analysis notebook
├── word2vec.ipynb               # Educational word2vec demo
├── The Well at the Worlds End.txt
├── DD_SRD_CC_v5.pdf
├── morris_descriptive.txt       # Extracted Morris passages
├── srd_descriptive.txt          # Extracted SRD passages
├── locations.txt                # AI-generated locations
├── structures.txt               # AI-generated structures
└── README.md
```

## Working with This Codebase

When modifying extraction thresholds:
1. Adjust constants at top of extractor file
2. Re-run extractor
3. Check console statistics for retention rate changes
4. Manually inspect sample passages to validate quality

When training new models:
1. Load extracted descriptive text (not raw source text)
2. Preprocess: tokenize → lemmatize → remove stopwords
3. Train with `min_count` appropriate to corpus size (20 for Morris, may need adjustment for smaller corpora)
4. Use `vector_size=300` for compatibility with pre-trained models

When comparing corpora:
1. Train separate models on each corpus
2. Extract vocabularies and embeddings
3. Use cosine similarity for word-level comparison
4. Use PCA + clustering for corpus-level visualization
5. Calculate distribution statistics (mean similarity, vocabulary overlap)
