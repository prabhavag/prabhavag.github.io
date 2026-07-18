---
title: "Qwen3-Omni (technical report)"
type: source
source_type: paper
developer: "[[Qwen Team]]"
published: 2025-09-22
arxiv: "2509.17765"
tags:
  - source
  - paper
  - multimodal
  - audio
  - speech
sources: 1
---

Technical report (arXiv [2509.17765](https://arxiv.org/pdf/2509.17765), Sept 2025) for
**[[Qwen3-Omni]]**, [[Qwen Team]]'s natively end-to-end **omni-modal** model — text, image,
audio, and video **in**, streaming text **or** speech **out**.

![[qwen3omni-fig1-capabilities.png]]
> *Figure 1 — Qwen3-Omni is a unified end-to-end model over text, audio, image and video that
> generates real-time text or speech; it supports voice dialogue, video dialogue, and video
> reasoning.*

## Why it matters here

Qwen3-Omni is a fourth data point for [[Real-Time Interactive Speech Models]], and the first
that is **omni-modal** rather than speech-centric. Its headline claim is **non-degradation**:
joint multimodal training reaches parity with same-size *unimodal* Qwen models on text and
vision while adding strong audio — refuting the usual "modality tax." For this wiki's audio
lens, three contributions matter:

1. **[[AuT (Audio Transformer)]]** — a from-scratch audio encoder (20M hours) that replaces
   Whisper, emitting general-purpose audio tokens at **12.5 Hz** with block-wise window
   attention for real-time prefill caching.
2. **[[Thinker-Talker Architecture]]** — a text-generating Thinker + a speech-token-generating
   Talker, both upgraded to **MoE**, with the Talker producing **multi-codebook RVQ** speech
   and a lightweight **Code2Wav ConvNet** vocoder for single-frame synthesis.
3. **Streaming**: end-to-end first-packet latency as low as **234 ms** (audio) / 547 ms
   (audio-video).

## Audio takeaways

- **Scale/results:** across **36 audio & audio-visual benchmarks**, open-source SOTA on 32 and
  overall SOTA on 22, beating Gemini 2.5 Pro, Seed-ASR, and GPT-4o-Transcribe on parts.
- **Long audio:** understands inputs **> 40 minutes**.
- **Languages:** 119 written; **19** spoken languages for understanding, **10** for generation.
- **Thinking model:** a reasoning variant covers full-modality (audio-only and audio-video)
  reasoning.
- **Sizes** (Qwen3-Omni-30B-A3B): AuT audio encoder 650M · SigLIP2 vision encoder 540M ·
  Thinker MoE 30B-A3B · Talker MoE 3B-A0.3B · MTP 80M · Code2Wav ConvNet 200M.

## Relation to existing sources

- vs **[[Moshi]]** — both are real-time speech-capable and both generate speech from **RVQ
  codec tokens**, but Moshi is a *single-stream-per-speaker* [[Multi-Stream Audio Modeling]]
  system with [[Inner Monologue]], while Qwen3-Omni splits generation into
  [[Thinker-Talker Architecture]] (text model + speech model). See the comparison on
  [[Real-Time Interactive Speech Models]].
- vs **[[Voxtral]]** — Voxtral is audio-**understanding** only ([[Speech Understanding]],
  audio→text). Qwen3-Omni both understands (its AuT encoder path is the analog of Voxtral's
  Whisper+adapter) **and** generates speech.
- Both Qwen3-Omni and Moshi drive a **12.5 Hz** codec rate — a recurring number for
  real-time speech token modeling (cf. [[Mimi (neural audio codec)]]).

## Open threads

- How does AuT's supervised-from-scratch encoder compare head-to-head with Whisper-based
  encoders ([[Voxtral]]) on understanding?
- Talker conditions on Thinker's *multimodal features but not its text representations* — what
  does decoupling cost/buy vs Moshi's text-audio [[Inner Monologue]]?
- Multi-codebook RVQ + MTP + ConvNet vocoder vs Moshi's [[RQ-Transformer]] Depth Transformer —
  which streaming speech-generation stack wins on latency/quality?
