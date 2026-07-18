---
title: "Voxtral"
type: entity
entity_type: model
developer: "[[Mistral AI]]"
tags:
  - entity
  - model
  - speech
  - audio-understanding
  - multimodal
  - open-weights
sources: 1
---

A pair of **open-weights (Apache 2.0) multimodal audio-chat models** by [[Mistral AI]]
(July 2025): **Voxtral Mini (4.7B)** and **Voxtral Small (24.3B)**. Voxtral takes **spoken
audio and text in, and produces text out** — it *comprehends* audio (transcribe, translate,
answer questions, summarize, call functions) rather than generating speech. State-of-the-art
on transcription/translation among open **and** closed models in its price class, while
preserving the text ability of its LLM backbone. A **32K context window** handles audio up to
**~40 minutes**.

This makes Voxtral a different class of model from the wiki's full-duplex dialogue cluster
([[Moshi]], [[Interaction Models]]): it is a **[[Speech Understanding]]** model (audio→text),
not a real-time speech-to-speech system. See [[Speech Understanding]] for the paradigm contrast.

## Architecture

![[voxtral-fig1-architecture.png]]
> *Figure 1 — Voxtral Architecture. The audio encoder attends to 30-second chunks of audio
> independently; the embeddings are concatenated and downsampled 4× in the audio-language
> adapter; the multimodal LLM decoder auto-regressively predicts text, conditioned on audio +
> text inputs.*

Three components, a conventional **encoder → adapter → decoder** cascade — the **opposite**
design bet to TML's [[Encoder-Free Early Fusion]]:

- **Audio encoder** — **[[Whisper large-v3]]** (640M). Raw waveform → 128-bin log-Mel
  spectrogram (160 hop) → conv stem (2× temporal downsample) → bidirectional self-attention →
  **50 Hz** audio embeddings. Whisper's fixed **30-second receptive field** is handled by
  encoding each 30 s chunk independently (positional encodings reset per chunk, chunks batched)
  — functionally chunk-wise attention, which controls cost and improves length generalization.
  Short audio is **padded** to 30 s (the [[#To pad or not to pad|padding ablation]] kept this).
- **Audio-language adapter** — an MLP that **downsamples 4×** (50 Hz → **12.5 Hz**), cutting the
  decoder sequence length. At 12.5 Hz a 30-min audio is ~22.5K tokens instead of 90K. (Note: the
  same **12.5 Hz** rate [[Moshi]]'s [[Mimi (neural audio codec)|Mimi]] codec lands on — Voxtral
  argues each audio embedding then carries ≈ one text-embedding's worth of information.)
- **Language decoder** — a Mistral LLM. **Mini** = [[Ministral 3B]] (edge-focused); **Small** =
  [[Mistral Small 3.1]] 24B. Audio is injected as embeddings; the decoder outputs text.

### Parameter counts (Table 1)

| Variant | Audio Encoder | Audio Adapter | Text Embeddings | Language Decoder | **Total** |
|---|---|---|---|---|---|
| **Mini** | 640M | 25M | 400M | 3.6B | **4.7B** |
| **Small** | 640M | 52M | 670M | 22.9B | **24.3B** |

## Training

Three phases — see [[Audio-Text Pretraining Patterns]] for the pretraining design, the key
contribution.

1. **Pretraining** — introduce speech to the text decoder. Audio is chunked into
   `(audio, transcript)` pairs (boundaries from VAD + diarization; missing transcripts are
   **ASR pseudo-labeled**). Two interleaving patterns sampled **50/50** —
   **audio-to-text repetition** (`<repeat>`, drives transcription) and **cross-modal
   continuation** (`<next>`, drives understanding) — plus text-only data to keep text skills.
   First pass **freezes encoder + decoder, trains only the adapter** (a warm-up that helps
   understanding evals). **Voxtral Mini Transcribe** is a repetition-only ASR variant.
2. **Supervised finetuning (SFT)** — mostly **synthetic**: long-form transcripts → Mistral Large
   generates QA / summarization / translation pairs *framed as if from listening*; text SFT
   (incl. **function-calling**) → converted to audio via TTS; real ASR questions answerable from
   world knowledge → paired with Mistral-Large answers (to fix TTS-only's poor generalization to
   real, accented speech). A special **"transcribe mode"** token removes the need for a text
   prompt on pure ASR.
3. **Preference alignment** — **DPO** and **[[Online DPO]]**. Clever trick: responses are ranked
   by a **text reward model** fed the audio's *transcription* (semantics/style/coherence
   transfer). Online DPO reused the **Magistral** sampling/reward infra.

## Results

State-of-the-art transcription + translation; competitive with GPT-4o mini / Gemini 2.5 Flash
on understanding.

![[voxtral-fig3-asr.png]]
> *Figure 3 — Speech Recognition (macro-avg WER ↓). Voxtral Small beats all open & closed
> models on English Short-Form and MCV; Voxtral Mini Transcribe beats GPT-4o mini Transcribe and
> Gemini 2.5 Flash on every task.*

- **Speech recognition (WER ↓):** Voxtral Small is SOTA on English Short-Form & MCV. Voxtral Mini
  Transcribe beats GPT-4o mini Transcribe and Gemini 2.5 Flash across all four task groups —
  while being a 4.7B model. (Per-task / per-language: Tables 3–6 in [[Voxtral (Mistral paper)]].)
- **Speech translation (BLEU ↑, FLEURS):** Voxtral Small is **SOTA on every source/target pair**
  tested, beating GPT-4o mini Audio and Gemini 2.5 Flash.
- **Speech understanding (Table 8):** competitive with closed models; Voxtral Small beats GPT-4o
  mini Audio on 3 of 7 tasks.
- **Text-only:** Voxtral Small matches [[Mistral Small 3.1]] across five text benchmarks — a
  **drop-in replacement** for both text and audio, i.e. audio is added *without* a text tax.

### Speech understanding accuracy (Table 8, %)

| Model | Llama QA | Openbook QA | MMLU* | MMAU* | Trivia QA* | GSM8k* | AU Bench |
|---|---|---|---|---|---|---|---|
| GPT-4o mini Audio | **74.3** | 83.7 | 72.6 | 63.4 | 83.7 | 90.8 | 80.0 |
| Gemini 2.5 Flash | 66.3 | **94.7** | **84.8** | **64.3** | **83.9** | **94.2** | **88.6** |
| **Voxtral Mini** | 54.3 | 59.6 | 47.6 | 57.1 | 54.9 | 71.6 | 85.6 |
| **Voxtral Small** | 71.7 | 88.4 | 74.3 | 62.2 | 79.4 | 89.7 | 86.6 |

`*` = speech-synthesized subset of a text benchmark (Voxtral's contributed evals). AU Bench =
the in-house Speech Understanding benchmark.

### Speech translation (Table 7, FLEURS BLEU ↑)

| Model | en→de | en→es | en→fr | en→it | de→en | es→en | fr→en | it→en |
|---|---|---|---|---|---|---|---|---|
| Whisper large-v3 | – | – | – | – | 46.1 | 34.9 | 43.0 | 35.7 |
| GPT-4o mini Audio | 44.5 | 36.5 | 52.7 | 37.3 | 51.8 | 41.6 | 48.2 | 41.5 |
| Gemini 2.5 Flash | 44.6 | 36.3 | 53.9 | 37.3 | 39.4 | 32.9 | 42.0 | 31.8 |
| **Voxtral Small** | **47.0** | **39.9** | **57.3** | **39.9** | **56.6** | **46.3** | **54.2** | **46.8** |

## Contributed benchmarks

To address the field's thin understanding-evals, Voxtral releases **speech-synthesized versions
of GSM8K, TriviaQA, and MMLU** (text benchmarks filtered to "verbalizable" prompts, TTS'd with
varied speakers) plus an internal **Speech Understanding (SU) benchmark** (≤19-min audios,
LLM-judge binary helpfulness + 0–5 grade). Released under a permissive license.

## Ablations (highlights)

See [[Voxtral (Mistral paper)]] for the figures.

- **To pad or not to pad** (§5.1): removing Whisper's 30 s padding gives ~no FLEURS-en penalty
  but 0.5% WER worse on French → **keep padding**.
- **Adapter downsampling** (§5.2): 12.5 Hz (4×) is the sweet spot — little ASR loss vs 50 Hz, and
  Llama QA actually **+1.5%** over the 50 Hz baseline; 6.25 Hz (8×) costs >1% WER. Hypothesis:
  at 12.5 Hz an audio embedding ≈ a text embedding's information content.
- **Pretrain patterns** (§5.3): repetition-only → strong ASR but ~0 on Llama QA; continuation-only
  → strong QA but ~60% WER; **50/50 recovers both**. → [[Audio-Text Pretraining Patterns]].
- **Online DPO** (§5.4, Table 2): improves SU response quality for both sizes; shipped as the
  default **Mini** checkpoint; **Small** default stays SFT (Online DPO slightly regressed English
  short-form WER).

## Relation to the rest of the wiki

- **Different paradigm from [[Moshi]] / [[Interaction Models]].** Those are **real-time
  full-duplex speech↔speech**; Voxtral is **audio→text understanding** (no speech output, not
  real-time-dialogue). → [[Speech Understanding]].
- **Encoder vs encoder-free.** Voxtral commits to a **[[Whisper large-v3]] encoder + adapter**;
  TML's [[Encoder-Free Early Fusion]] deliberately avoids standalone encoders. [[Moshi]] takes a
  third route (a learned **codec**, [[Mimi (neural audio codec)]]).
- **Shared tricks with [[Moshi]]:** ASR **pseudo-labeling** of untranscribed audio (Whisper),
  and a **12.5 Hz** audio frame rate.

Detailed in source: [[Voxtral (Mistral paper)]].
