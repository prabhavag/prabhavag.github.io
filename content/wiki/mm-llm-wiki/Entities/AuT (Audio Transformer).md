---
title: "AuT (Audio Transformer)"
type: entity
entity_type: model
developer: "[[Qwen Team]]"
tags:
  - entity
  - model
  - audio
  - speech
  - audio-encoder
sources: 1
---

The **audio encoder** of [[Qwen3-Omni]] — an attention-based **encoder–decoder,
auto-regressive** model trained **from scratch on 20 million hours of supervised audio**. It
replaces the Whisper encoder used in Qwen2.5-Omni (and in [[Voxtral]]) with a purpose-built,
general-purpose audio representation. ~**0.6B / 650M parameters**. Source:
[[Qwen3-Omni (technical report)]].

![[qwen3omni-fig3-aut-encoder.png]]
> *Figure 3 — AuT overview. An attention encoder–decoder auto-regressive model trained from
> scratch on 20M hours of supervised audio; Qwen3-Omni uses the **encoder** to obtain
> general-purpose audio representations at a **12.5 Hz** token rate.*

## How it works

- **Input:** filter-bank / mel features → **Conv2D blocks downsample 8×** before the attention
  layers → token rate **12.5 Hz** (~**80 ms** per frame). (In Qwen3-Omni the front end is a
  128-channel mel-spectrogram at 16 kHz, 25 ms window / 10 ms hop.)
- **Training tasks:** speech recognition **and** audio understanding, for stronger and more
  general representations than ASR-only encoders. Data mix: **80%** Chinese/English
  pseudo-labeled ASR · **10%** other-language ASR · **10%** audio understanding.
- **Streaming trick — block-wise window attention:** AuT uses **flash attention with dynamic
  attention window sizes** (query patterns from **1 to 8 seconds**), balancing **real-time
  prefill caching** against performance on offline/long-audio tasks. This is what lets the
  encoder participate in ~234 ms end-to-end streaming.
- In [[Qwen3-Omni]] only the **encoder** is used (the decoder is a training-time device).

## Why it's notable

- **From-scratch, supervised, 20M-hour** encoder is a bet that a bespoke audio backbone beats
  adapting Whisper — a direct contrast with [[Voxtral]], which keeps a **Whisper large-v3**
  encoder + adapter for [[Speech Understanding]].
- The **12.5 Hz** output rate matches the codec rate used elsewhere in real-time speech systems
  (cf. [[Mimi (neural audio codec)]] in [[Moshi]]) — a convergent "one token ≈ 80 ms" design
  point that keeps decoder sequence lengths tractable for long audio and low latency.

## See also

- [[Qwen3-Omni]] · [[Thinker-Talker Architecture]]
- Encoder contrast: [[Voxtral]] (Whisper-based) · rate contrast: [[Mimi (neural audio codec)]]
