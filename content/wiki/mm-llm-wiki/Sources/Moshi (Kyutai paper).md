---
title: "Moshi: a speech-text foundation model for real-time dialogue"
type: source
source: "https://arxiv.org/pdf/2410.00037"
author:
  - "[[Kyutai]]"
published: 2024-09-17
created: 2026-06-06
tags:
  - source
  - speech
  - full-duplex
  - real-time
  - multimodal
---

**Source clip:** [[Moshi - a speech-text foundation model for real-time dialogue]] · [[Kyutai]] · arXiv 2410.00037 (Sept 2024)

Technical report for **[[Moshi]]**, the first real-time **[[Full-Duplex Spoken Dialogue|full-duplex]]**
speech-text dialogue model. Moshi casts spoken dialogue as **speech-to-speech generation**,
eliminating the cascaded ASR → LLM → TTS pipeline and its compounding latency, text
bottleneck, and rigid speaker turns. Theoretical latency **160 ms** (200 ms in practice),
below the ~230 ms human average. Open-sourced at `github.com/kyutai-labs/moshi`.

![[moshi-fig1-overview.png]]
> *Figure 1 — Overview of Moshi.*

> **Figures & tables** (the PDF→markdown conversion dropped all images and mangled tables, so
> these were restored manually): architecture figures rendered from the PDF live on [[Moshi]]
> (Fig 1), [[Mimi (neural audio codec)]] (Fig 2), [[RQ-Transformer]] (Fig 3),
> [[Multi-Stream Audio Modeling]] (Fig 4). Clean results tables: text-LM eval on [[Helium]],
> spoken-QA on [[Moshi]].

## Key takeaways

- **The cascaded-pipeline problem.** Conventional voice assistants chain voice-activity
  detection, ASR, a text LLM, and TTS. This yields multi-second latency, discards
  paralinguistic information (emotion, non-speech audio) because text is the bottleneck, and
  forces a turn-based model of dialogue that cannot represent overlap, interruptions, or
  backchanneling (10–20% of spoken time is overlap).
- **Three stacked components:**
  - **[[Helium]]** — a 7B text LLM trained from scratch on 2.1T tokens of filtered English,
    supplying reasoning/knowledge.
  - **[[Mimi (neural audio codec)|Mimi]]** — a streaming neural audio codec (12.5 Hz, 1.1 kbps)
    that fuses **semantic + acoustic** tokens via a **split RVQ** with WavLM distillation.
  - **[[RQ-Transformer]]** — a Temporal Transformer (over time) + Depth Transformer (over the
    codebooks within a step) that models the token hierarchy in a streaming fashion.
- **[[Multi-Stream Audio Modeling]]:** Moshi models **two audio streams in parallel** (its own
  and the user's) as joint token streams, so there are **no explicit speaker turns** — it
  always listens and always generates (speech or silence). This is what makes it full-duplex.
- **[[Inner Monologue]]:** Moshi predicts **time-aligned text tokens as a prefix** to its
  audio tokens (text → semantic → acoustic). This is the single most impactful design choice —
  it roughly **triples spoken-QA accuracy** at ~no inference cost, and by changing the
  text↔audio **delay** the same model becomes a streaming **ASR** or **TTS** system.
- **Joint sequence:** K = 2Q+1 = **17 sub-streams** (1 text + 8 Moshi audio + 8 user audio),
  Q = 8 codebooks at 12.5 Hz.

## Results

- **Helium** is on-par with or beats similarly-compute-budgeted 7B LLMs (MPT, Falcon, Llama 2,
  OLMo) on ARC/OBQA/HellaSwag/MMLU etc.
- **Spoken QA** (Web Questions, Llama Questions, audio TriviaQA): Moshi is the best
  speech-to-speech model; Inner Monologue ~triples accuracy vs audio-only.
- **Streaming TTS** 4.7% WER on LibriSpeech test-clean (beats Vall-E 5.9%; behind
  NaturalSpeech 3 at 1.81% but with only 2 s lookahead). **Streaming ASR** 5.7% WER with 80 ms
  alignment precision.
- **Generated dialogues** match a cascaded ASR+LM+TTS topline on linguistic quality while
  producing realistic turn-taking (pauses, gaps, overlaps).
- **Cost of audio training:** MMLU drops 54.3 (Helium) → **49.7** (Moshi); the largest QA
  regressions trace to oral-style fine-tuning, not lost knowledge.

## Safety

Toxicity analysis, training-data regurgitation analysis, system **voice consistency** (so the
model keeps a fixed identity), and **watermarking** to identify Moshi-generated audio.

## Relation to the rest of the wiki

Moshi is a **direct predecessor** of [[Interaction Models]] / [[TML-Interaction-Small]] — the
TML blog explicitly cites "audio full-duplex models" as prior art. Key contrasts worth
tracking:

- **Tokenizer vs encoder-free.** Moshi routes audio through the **Mimi codec** (discrete
  tokens); TML uses **[[Encoder-Free Early Fusion]]** (dMel/hMLP, minimal preprocessing). →
  see [[Encoder-Free Early Fusion]].
- **Single model vs split.** Moshi is **one** streaming model; TML adds an async
  **[[Interaction-Background Model Split|background model]]** for heavy reasoning.
- **Continuous token streams vs micro-turns.** Moshi interleaves fixed-rate (12.5 Hz) parallel
  token streams; TML uses **[[Time-Aligned Micro-Turns]]** (200 ms chunks). Same goal — keeping
  silence/overlap/interruption in context — via different mechanisms.
- **Audio-only vs any modality.** Moshi is speech↔speech (+ text scaffolding); Interaction
  Models target audio + video + text.

## Open threads to pursue

- How would Moshi score on [[FD-bench]] and the TML proactivity benchmarks (TimeSpeak/CueSpeak)?
- Does the multi-stream token approach scale to **video**, or is the codec route a ceiling?
- Inner Monologue's text-prefix vs TML's text co-training — which generalizes better?
