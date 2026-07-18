---
title: "Interaction Models"
type: concept
tags:
  - concept
  - real-time
  - multimodal
  - human-ai-collaboration
sources: 1
---

# Interaction Models

Models that **handle interaction natively** rather than through external scaffolding. Instead
of consuming a complete user turn and emitting a complete response, an interaction model is
in constant two-way exchange — perceiving and responding at the same time, across audio,
video, and text.

## Why (the argument)

- Turn-based models "experience reality in a single thread": they can't perceive the user
  mid-turn, and freeze perception while generating. This is the **collaboration bottleneck**
  — a narrow channel that pushes humans out of the loop.
- Existing real-time systems emulate interactivity with a **harness** (voice-activity
  detection, dialog management, TTS) made of components *less* intelligent than the model.
- Per "the bitter lesson," hand-crafted harnesses get outpaced by general capability — so
  **interactivity must be part of the model itself**, scaling with intelligence.

## How it's realized

- [[Time-Aligned Micro-Turns]] — continuous streams split into interleaved 200ms chunks.
- [[Interaction-Background Model Split]] — real-time presence + async heavy reasoning.
- [[Encoder-Free Early Fusion]] — minimal-preprocessing multimodal input.

## Capabilities it unlocks

Seamless dialog management, verbal/visual interjections, simultaneous speech (live
translation), time-awareness, and concurrent tools/search/UI while listening and speaking.

### Demos

The capability demos (hero + 9 capability videos) are embedded in the source page's media
catalog — see [[Interaction Models (TML blog)#Demo media]].

## Instances & evidence

- [[TML-Interaction-Small]] by [[Thinking Machines Lab]] — first concrete model.
- Measured by [[FD-bench]] and proactivity benchmarks (TimeSpeak, CueSpeak, RepCount-A,
  ProactiveVideoQA, Charades).

## Prior & related work

The audio-only precursor is **[[Full-Duplex Spoken Dialogue]]**, exemplified by [[Moshi]]
([[Kyutai]], 2024) — cited in the TML blog as prior art. Moshi achieves real-time
interactivity via [[Multi-Stream Audio Modeling]] over a [[Mimi (neural audio codec)|codec]];
Interaction Models generalize the idea to audio + video + text and go encoder-free. See
[[Moshi (Kyutai paper)]].

Source: [[Interaction Models (TML blog)]].
