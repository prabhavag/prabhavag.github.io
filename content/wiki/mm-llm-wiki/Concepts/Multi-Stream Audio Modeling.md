---
title: "Multi-Stream Audio Modeling"
type: concept
tags:
  - concept
  - speech
  - architecture
  - real-time
sources: 1
---

# Multi-Stream Audio Modeling

The mechanism that makes [[Moshi]] **[[Full-Duplex Spoken Dialogue|full-duplex]]**: the model
jointly models **two audio streams in parallel** — its own output and the user's input — as
separate but simultaneously-modeled autoregressive token streams.

![[moshi-fig4-joint-sequence.png]]
> *Figure 4 — The joint sequence (acoustic delay τ=1). Top: the **user stream** (semantic +
> acoustic tokens, fed in). Below the dashed line: the **Moshi stream** — text tokens (with
> `PAD`/`EPAD`, see [[Inner Monologue]]), then semantic, then acoustic tokens (sampled).*

## How it works

- Each speaker's audio is tokenized by [[Mimi (neural audio codec)|Mimi]] into Q = 8 codebooks
  at 12.5 Hz; both streams (Moshi's `A` and the user's `A′`) are concatenated into the joint
  sequence modeled by the [[RQ-Transformer]].
- With the text stream from [[Inner Monologue]], the joint target has **K = 2Q+1 = 17**
  sub-streams (1 text + 8 Moshi audio + 8 user audio).
- At inference Moshi **samples its own** tokens while the **user's** stream is fed from the real
  microphone (the modeled user stream is only used to *simulate* dialogues offline).

## Why it matters

- **No explicit speaker turns.** Moshi can speak, listen, or do both at once; overlap,
  interruptions, and backchanneling fall out naturally instead of needing a turn-prediction
  harness (VAD).
- When the user speaks and Moshi is silent, Moshi's audio stream decodes to "natural silence"
  and its text stream fills with `PAD` — silence is modeled, not a special case.

## Relation to other approaches

This is the audio-domain analogue of TML's **[[Time-Aligned Micro-Turns]]**: both keep silence,
overlap, and interruption in context, but Moshi uses **fixed-rate parallel token streams** over
a [[Mimi (neural audio codec)|codec]], while TML interleaves **200 ms multimodal micro-turns**.
See [[Full-Duplex Spoken Dialogue]].

A different tack on the same problem is [[Qwen3-Omni]]'s [[Thinker-Talker Architecture]], which
**splits** text generation (Thinker) from speech-token generation (Talker) rather than running
parallel per-speaker streams — but still autoregresses over RVQ codec tokens at the same 12.5 Hz.

Source: [[Moshi (Kyutai paper)]].
