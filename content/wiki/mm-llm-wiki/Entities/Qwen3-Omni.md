---
title: "Qwen3-Omni"
type: entity
entity_type: model
developer: "[[Qwen Team]]"
tags:
  - entity
  - model
  - multimodal
  - omni-modal
  - audio
  - speech
  - full-duplex
  - open-weights
sources: 1
---

A natively **end-to-end omni-modal model** by [[Qwen Team]] (Sept 2025) that takes **text,
image, audio, and video in** and produces **streaming text or speech out**. The flagship is
**Qwen3-Omni-30B-A3B** (a Mixture-of-Experts model, 30B total / 3B active). Source:
[[Qwen3-Omni (technical report)]].

Its central claim is **non-degradation**: with joint multimodal training (mixing unimodal +
cross-modal data early in text pretraining), it matches same-size *unimodal* Qwen text and
vision models **while** adding strong audio — no modality tax. Across 36 audio / audio-visual
benchmarks it reports open-source SOTA on 32 and overall SOTA on 22.

For this wiki's focus it belongs to the [[Real-Time Interactive Speech Models]] cluster
alongside [[Moshi]] and [[Interaction Models]] — but unlike them it is omni-modal, and unlike
the understanding-only [[Voxtral]] it both understands **and generates** speech.

## Architecture — [[Thinker-Talker Architecture]]

![[qwen3omni-fig2-thinker-talker.png]]
> *Figure 2 — Qwen3-Omni's Thinker-Talker architecture. The **Thinker** generates text; the
> **Talker** generates streaming speech tokens from the Thinker's multimodal representations.
> The Talker autoregressively predicts a multi-codebook sequence; per step an **MTP** module
> emits the residual codebooks and **Code2Wav** incrementally renders the waveform,
> frame-by-frame.*

Component sizes (30B-A3B):

| Module | Architecture | Params | Streaming |
|---|---|---|---|
| Audio encoder | [[AuT (Audio Transformer)]] | 650M | ✓ |
| Vision encoder | SigLIP2-So400M | 540M | – |
| Thinker | MoE Transformer | 30B-A3B | ✓ |
| Talker | MoE Transformer | 3B-A0.3B | ✓ |
| MTP | Dense Transformer | 80M | ✓ |
| Code2Wav | ConvNet | 200M | ✓ |

End-to-end first-packet latency: **234 ms (audio) / 547 ms (audio-video)**.

## Audio path (input → understanding)

- Audio is resampled to 16 kHz → 128-channel mel-spectrogram (25 ms window, 10 ms hop) →
  **[[AuT (Audio Transformer)]]** encoder → tokens at **12.5 Hz** (~80 ms per frame).
- Position handled by **TM-RoPE** (Time-aligned Multimodal RoPE): temporal/height/width angles
  split 24/20/20; audio gets one temporal ID per **80 ms**, anchored to absolute time so
  audiovisual streams of arbitrary length align without the fixed 2-second chunking used in
  Qwen2.5-Omni.
- Handles audio **> 40 minutes**; 19 spoken languages understood.

## Speech generation (Talker)

- The **Talker** conditions on the Thinker's **multimodal features but not its text
  representations** (discrete text tokens ≈ embeddings; decoupling lets external modules —
  RAG, function-calling, safety — intervene on text before synthesis).
- Operates directly on **RVQ (multi-codebook) tokens**: backbone predicts codebook 0 via a
  linear head; an **MTP** module predicts all residual codebooks for the frame.
- **Code2Wav** is a lightweight **causal ConvNet** vocoder (replaces the block-wise DiT of
  Qwen2.5-Omni) → lower latency/FLOPs, single-frame immediate synthesis at **12.5 Hz**.
- 10 spoken languages for generation; multi-codebook representation targets diverse voices and
  paralinguistic cues.

## Upgrades over Qwen2.5-Omni

Thinker & Talker → **MoE**; Whisper → **[[AuT (Audio Transformer)]]**; single-track → **multi-track
multi-codebook** codec with MTP; DiT vocoder → **Code2Wav ConvNet**; input/output rates → **12.5
Hz** with single-frame synthesis; plus long-audio (>40 min), 119 written languages, a
**Thinking** (reasoning) variant, and ~234 ms streaming latency.

## See also

- [[AuT (Audio Transformer)]] · [[Thinker-Talker Architecture]]
- [[Real-Time Interactive Speech Models]] — where it sits vs [[Moshi]] / [[Interaction Models]] / [[Voxtral]]
- Contrasts: [[Multi-Stream Audio Modeling]] & [[Inner Monologue]] (Moshi), [[Speech Understanding]] (Voxtral)
