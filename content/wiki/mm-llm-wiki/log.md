# Log

Append-only record of wiki activity. Each entry starts with `## [YYYY-MM-DD] <op> | <subject>`
so the log stays grep-able: `grep "^## \[" log.md | tail -5`.

## [2026-06-02] ingest | Interaction Models (TML blog)
First source ingested; wiki seeded from empty. Established `Sources/`, `Entities/`, `Concepts/`
folder layout plus `index.md` and this log.
- Source summary: [[Interaction Models (TML blog)]] (raw clip in `Clippings/`).
- New entities: [[Thinking Machines Lab]], [[TML-Interaction-Small]].
- New concepts: [[Interaction Models]], [[Time-Aligned Micro-Turns]],
  [[Interaction-Background Model Split]], [[Encoder-Free Early Fusion]], [[FD-bench]].
- Open threads logged on the source page: background-agent comparison to other agentic
  architectures; benchmark reproducibility (some baselines self-reported).

## [2026-06-06] schema | Companies are not entities
Convention set: `Entities/` is for subjects of study (models, techniques, benchmarks), not
publishers. Deleted `Entities/Thinking Machines Lab.md`; the lab now lives only as
author/publisher `[[Thinking Machines Lab]]` wikilinks (left unresolved by design). Removed
its bullet from `index.md` (pages 8 → 7). Recorded the rule in `CLAUDE.md` → Conventions.

## [2026-06-06] media | Restored TML blog demo videos/animations
The web clipper dropped the page's demo media. Restored (link/reference only, no source text
reworded) into the raw clip and surfaced across the wiki: hero video, 9 capability demos
(YouTube), the 16-frame "Our approach" micro-turn animation, and the benchmark example clips
(5 audio comparison sets + example-5 mp4). User-authorized augmentation of `raw/`.
- Added `## Demo media` catalog to [[Interaction Models (TML blog)]].
- Embedded/linked demos on [[Interaction Models]], [[Time-Aligned Micro-Turns]],
  [[Interaction-Background Model Split]], [[FD-bench]], [[TML-Interaction-Small]].
- **Vault quirk found:** an Obsidian plugin auto-localizes `![](url)` image embeds — it saved
  YouTube watch pages as 1 MB HTML `.jpg` files (deleted). Convention: use plain `[label](url)`
  links for video/audio; only real image URLs survive embedding. Recorded in `CLAUDE.md`.
- Real frame stills downloaded by the localizer kept at `raw/assets/{1,6,12,16}.jpg`.

## [2026-06-06] media | Restored the 4 "Our approach" figures
The clipper dropped/flattened the four diagram figures. They are CSS/SVG (no image files
except Fig 1's frames), so reproduced as Mermaid/structured diagrams + captions:
- Fig 1 (turn-based vs. micro-turn timeline) → frames + caption on [[Time-Aligned Micro-Turns]].
- Fig 2 (system overview) → Mermaid on [[Interaction-Background Model Split]] and raw clip
  (was entirely absent).
- Fig 3 (human perception vs. interleaved token sequence) → reconstructed diagram on
  [[Time-Aligned Micro-Turns]] and in the raw clip (clipper had flattened it to loose labels).
- Fig 4 (single-micro-turn architecture) → Mermaid on [[Encoder-Free Early Fusion]]; source's
  inline SVG preserved in the raw clip.
- Figure index added to the `## Demo media` section of [[Interaction Models (TML blog)]].

## [2026-06-06] media | Replaced figure reproductions with real screenshots
Per request, captured screenshots of the original four diagrams (they are JS/CSS/SVG, not
downloadable images) using Playwright driving system Chrome (`channel: 'chrome'`). Hid the
animation control buttons + hover-cue and let the animated figures settle before capture.
Saved to `raw/assets/fig{1..4}-*.png` (≈2× DPI) and embedded them in place of the earlier
Mermaid/structured reproductions on [[Time-Aligned Micro-Turns]] (Fig 1 + 3),
[[Interaction-Background Model Split]] (Fig 2), [[Encoder-Free Early Fusion]] (Fig 4), and in
the raw clip. Deleted the now-orphaned `raw/assets/{1,6,12,16}.jpg` frame stills. Fig 4's
original inline SVG remains in the raw clip.

## [2026-06-06] media | Embed videos as players (iframes) instead of links
Replaced every `[▶ …](youtube)` link with an inline HTML `<iframe src=".../embed/<ID>">`
player, and the self-hosted `example-5/video.mp4` with a `<video controls>` tag. iframes/
`<video>` are raw HTML, which the image-localizer leaves alone (verified — no junk created),
unlike `![](url)`. Embedded on [[Interaction Models]] (hero + 9 demos), [[Time-Aligned Micro-Turns]]
(time awareness), [[Interaction-Background Model Split]] (2 demos), [[TML-Interaction-Small]]
(hero), [[FD-bench]] (benchmark mp4), the `## Demo media` catalog on [[Interaction Models (TML blog)]],
and the raw clip. Updated the media convention in `CLAUDE.md`.

## [2026-06-06] lint | Reclassify FD-bench as Entity
Moved `Concepts/FD-bench.md` → `Entities/FD-bench.md` (benchmarks are entities per schema:
"things the wiki actually analyzes — models, products, techniques, benchmarks, datasets").
Updated frontmatter (`type: entity`, `entity_type: benchmark`) and re-filed under Entities in
`index.md`. All `[[FD-bench]]` wikilinks unaffected (Obsidian resolves by filename).

## [2026-06-06] ingest | Moshi (Kyutai paper)
Second source ingested (arXiv 2410.00037, PDF → markdown via pymupdf4llm into `raw/`, frontmatter
added). Moshi is the first real-time full-duplex speech-text dialogue model — a direct
predecessor to the existing TML [[Interaction Models]] cluster.
- Source summary: [[Moshi (Kyutai paper)]] (raw clip in `raw/`).
- New entities: [[Moshi]], [[Mimi (neural audio codec)]], [[Helium]].
- New concepts: [[Full-Duplex Spoken Dialogue]], [[Multi-Stream Audio Modeling]],
  [[Inner Monologue]], [[RQ-Transformer]].
- New topic (first one!): [[Real-Time Interactive Speech Models]] — synthesis comparing Moshi
  vs Interaction Models (codec vs encoder-free; single-model vs background split; token streams
  vs micro-turns).
- Cross-linked into existing pages: [[Interaction Models]] (prior work), [[Time-Aligned Micro-Turns]],
  [[Encoder-Free Early Fusion]], [[Interaction-Background Model Split]], [[TML-Interaction-Small]],
  [[FD-bench]]. Author [[Kyutai]] kept as publisher wikilink only (no entity page, per schema).
- Open threads: evaluate Moshi on FD-bench; Inner Monologue vs text co-training; codec ceiling
  vs encoder-free for video.

## [2026-06-06] media | Restore Moshi figures & tables (initial ingest had none)
pymupdf4llm omitted all paper images ("picture intentionally omitted") and mangled tables
(multi-value cells with `<br>`). Fixed:
- Rendered the 4 architecture figures directly from the PDF (PyMuPDF, crop each figure region
  above its caption, 3× zoom) → `raw/assets/moshi-fig{1..4}-*.png`. Embedded Fig 1 on [[Moshi]],
  Fig 2 on [[Mimi (neural audio codec)]], Fig 3 on [[RQ-Transformer]], Fig 4 on
  [[Multi-Stream Audio Modeling]] (referenced from [[Inner Monologue]]); replaced the omitted-image
  placeholders in the raw clip. (Fig 4's crop needed a manual top because its internal labels
  read as preceding text.)
- Reconstructed key tables as clean markdown: Table 2 (text-LM eval) on [[Helium]], Table 8
  (spoken QA) on [[Moshi]].
- Fixed a broken markdown table on [[Real-Time Interactive Speech Models]] — a linter had split
  cells on the `|` inside `[[link|alias]]`; rewrote without alias pipes.

## [2026-06-06] media | Complete the Moshi raw page + codify PDF figure/table handling
Finished restoring the raw clip so it mirrors the paper: rendered the remaining figures
(Fig 5–11) and the two mangled tables (Table 1 hyperparameters, Table 3 Mimi ablation) from the
PDF via PyMuPDF and embedded them, replacing the `intentionally omitted` placeholders. Figures
bounded by drawings/text union above each caption; tables bounded by their horizontal rule
lines. The 7 remaining placeholders are inline equations (no caption) — left as-is. New assets:
`raw/assets/moshi-fig{5..11}.png`, `moshi-table1-hyperparams.png`, `moshi-table3-mimi-ablation.png`.
Updated **CLAUDE.md** → PDF ingestion workflow with a reusable figure/table rendering recipe
(venv install, caption-based crop, rule-line table bounds, equation placeholders left alone,
markdown-table `|`-in-wikilink gotcha) so future PDF ingests bring figures/tables in by default.

## [2026-06-06] query | How is real-world data used to train time-aligned micro-turn models (TML-small)?
Finding: the [[Interaction Models (TML blog)]] does **not** disclose TML-small's training corpus
(only "trained from scratch", "improves with model size + training data", and synthetic safety
data). Answered using [[Moshi]]'s documented recipe as the analogue: 7M h unlabeled real audio
(Whisper-transcribed) → Fisher 2000h real two-channel calls for full-duplex → PyAnnote
diarization to synthesize multi-stream from single-channel audio → 170h real multi-channel
fine-tune → 20k h synthetic instruct grounded in Wikipedia/StackExchange. Per user preference,
**no standalone page** — folded the answer into the relevant entities: "Training data" sections
on [[Moshi]] (the documented recipe) and [[TML-Interaction-Small]] (undisclosed), with a pointer
from [[Time-Aligned Micro-Turns]].

## [2026-06-06] ingest | Voxtral (Mistral paper)
Third source ingested (arXiv 2507.13264, July 2025; PDF → markdown via pymupdf4llm into `raw/`,
frontmatter added). Voxtral is the wiki's **first speech-*understanding* model** (audio→text),
a sibling paradigm to the existing real-time full-duplex cluster rather than a member of it.
- Source summary: [[Voxtral (Mistral paper)]]; raw clip [[Voxtral]].
- New entity: [[Voxtral]] (Mini 4.7B / Small 24.3B; Whisper encoder → 4× adapter → Mistral LLM).
- New concepts: [[Speech Understanding]] (the audio-in/text-out paradigm + the encoder→adapter→LLM
  cascade, contrasted with full-duplex speech-to-speech) and [[Audio-Text Pretraining Patterns]]
  (`<repeat>` repetition + `<next>` cross-modal continuation; the 50/50 result).
- Cross-linked into existing pages: [[Moshi]] (12.5 Hz + ASR-pseudo-label convergences; codec vs
  encoder), [[Encoder-Free Early Fusion]] (added Voxtral as the encoder+adapter third design
  point), [[Inner Monologue]] (token- vs segment-level text coupling), [[Real-Time Interactive
  Speech Models]] (added a "sibling paradigm" section + Voxtral as a contrasting source).
  Backbones/baselines left as unresolved links by design: [[Mistral AI]] (publisher),
  [[Whisper large-v3]], [[Ministral 3B]], [[Mistral Small 3.1]], [[Online DPO]].
- Figures/tables: pymupdf4llm dropped all 9 figures ("picture intentionally omitted") and mangled
  Tables 2 & 3 (`<br>`-crammed cells). Rendered all 9 figures from the PDF (caption-below-figure
  crop, 3× zoom) → `raw/assets/voxtral-fig{1..9}-*.png`; embedded Fig 1/3 on [[Voxtral]], Fig 2 on
  [[Audio-Text Pretraining Patterns]], all 9 in the raw clip. Reconstructed Tables 2 (DPO) & 3
  (English ASR) as clean markdown in the raw clip; clean Tables 1/7/8 rebuilt on [[Voxtral]].
  Header logo banner placeholder left as-is (decorative). Tables 4–6 converted cleanly.
- Open threads (on the source page): can an encoder+adapter understanding stack go streaming/
  full-duplex; cost of a transcription-fed (audio-blind) text reward model in [[Online DPO]];
  how Voxtral's speech-synthesized benchmarks would rank the full-duplex models on understanding.

## [2026-07-11] media | Rebuild Voxtral from arXiv LaTeX source; add TeX-first extractor workflow

- Re-converted [[Voxtral]] from its **arXiv LaTeX source** (`arxiv.org/e-print/2507.13264`)
  instead of the PDF: `latexpand` (flatten `\input`) → `pandoc -t gfm --citeproc`. Result
  replaces `raw/Voxtral.md`. Gains over the old pymupdf4llm version: exact tables (Table 3's
  rotated Short-Form/Long-Form spanning header preserved as HTML `colspan`), verbatim math,
  resolved `(Author Year)` citations + reference list, and no OCR.
- Replaced all figure assets with the authors' **original figure files** from the source
  tarball (PNG, capped 2400px wide) at the same names, so every wiki embed upgraded with zero
  link changes: `raw/assets/voxtral-fig{1..9}-*.png` (e.g. fig3-asr 1284×542 → 2400×1492) plus
  new `voxtral-header.png` banner. Verified all 10 embeds resolve; wiki refs on [[Voxtral]]
  (Entity), [[Voxtral (Mistral paper)]] (Source), [[Audio-Text Pretraining Patterns]] intact.
- Also evaluated `marker-pdf` on the same PDF (deep-learning layout model): auto-extracted
  figures + good tables/math but low-res JPEG crops, OCR slips, and mangled the rotated header.
  TeX source beat it on every axis, so **marker was not adopted**; the package and its reference
  output were later removed.
- Toolchain: rebuilt `mlenv` as **native arm64** (was x86_64 under Rosetta — crashed torch).
  Now has `pandoc` + `pymupdf4llm`; `latexpand` at `~/.local/bin`. Codified the order in
  `CLAUDE.md`: **LaTeX source → pymupdf4llm**.

## [2026-07-11] media | Rebuild Moshi from arXiv LaTeX source (full clean re-ingest)

- Replaced `raw/Moshi - a speech-text foundation model for real-time dialogue.md` with a TeX-source
  conversion of arXiv 2410.00037 (`latexpand --expand-bbl` → `pandoc -t gfm`). Gains: 111 resolved
  author-year citations, real tables (incl. the former image-tables `tab:hparams` / `tab:mimi_ablations`),
  verbatim math, and section cross-refs resolved to titles.
- Citations: the source ships only a natbib `main.bbl` (no `.bib`), which pandoc/citeproc couldn't use,
  so I mapped the 117 `\bibitem[Author(Year)]{key}` labels myself and rewrote all 213 `\cite*` calls to
  author-year text before pandoc.
- Figures: rebuilt all 11 from the source figure files — F1–F3 PNG, F4/F8/F9 rendered from vector PDF,
  and the 5 multi-panel appendix figures (F5–F7, F10–F11) rendered per-panel and composited. Body figs
  keep names (`moshi-fig1-overview`..`fig4-joint-sequence`); appendix figs got descriptive names
  (`moshi-fig5-mosnet-scores` … `fig11-artifacts-summary`). All capped 2400px.
- Cleanup: removed the 7 old generic `moshi-fig5..11.png` crops and the 2 table images
  (`moshi-table1-hyperparams`, `moshi-table3-mimi-ablation`) now that tables are real markup. Wiki refs
  on [[Moshi]], [[Mimi (neural audio codec)]], [[RQ-Transformer]], [[Multi-Stream Audio Modeling]],
  [[Moshi (Kyutai paper)]] verified intact (fig1–4 names unchanged).

## [2026-07-11] ingest | Qwen3-Omni Technical Report

Fourth source (arXiv 2509.17765, Sept 2025), ingested via the TeX-source workflow
(`latexpand` → `pandoc --citeproc` with `biblio.bib`; all citations resolved, 0 not-found).
Raw doc: [[Qwen3-Omni Technical Report]] (748 lines, 3 figures, 7 complex tables kept as HTML).
Per request, full source ingested but **wiki synthesis focuses on the audio part**.
- Figures rendered from source: `qwen3omni-fig1-capabilities` (from vector PDF),
  `-fig2-thinker-talker`, `-fig3-aut-encoder` (PNG), capped 2400px.
- Table cross-refs resolved by true `\begin{table}` order (label-order was wrong — some tables
  are unlabeled; e.g. inference-latency = Table 2, non-degradation = Table 16).
- New source: [[Qwen3-Omni (technical report)]]. New entities: [[Qwen3-Omni]] (omni-modal
  30B-A3B MoE, Thinker-Talker, ~234ms), [[AuT (Audio Transformer)]] (from-scratch 20M-hr audio
  encoder, 12.5Hz, block-wise window attention, replaces Whisper). New concept:
  [[Thinker-Talker Architecture]] (Thinker text + Talker speech; MTP + Code2Wav ConvNet vocoder).
- Integrated into [[Real-Time Interactive Speech Models]] as a 3rd real-time approach (added a
  Qwen3-Omni column to the comparison table); cross-linked from [[Speech Understanding]]
  (AuT vs Whisper audio-in path) and [[Multi-Stream Audio Modeling]] (Thinker-Talker vs parallel
  streams). Author [[Qwen Team]] kept as publisher wikilink only (no entity page, per schema).
- Notable: Moshi and Qwen3-Omni converge on RVQ codec speech at **12.5 Hz** despite very
  different architectures. Open threads on the source page: AuT vs Whisper encoders; cost/benefit
  of decoupling Talker from Thinker text; multi-codebook RVQ+MTP+ConvNet vs [[RQ-Transformer]].
