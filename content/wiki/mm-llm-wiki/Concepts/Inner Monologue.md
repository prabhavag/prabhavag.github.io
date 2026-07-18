---
title: "Inner Monologue"
type: concept
tags:
  - concept
  - speech
  - architecture
sources: 1
---

# Inner Monologue

A training/inference technique from [[Moshi]]: the model predicts **time-aligned text tokens
as a prefix to its audio tokens** at every step (text → semantic → acoustic). The text acts as
"scaffolding" that lets a speech-to-speech model exploit the linguistic competence of its text
backbone ([[Helium]]) without leaving the audio domain.

The text/`PAD`/`EPAD` stream is the bottom row of Moshi's joint sequence — see Figure 4 on
[[Multi-Stream Audio Modeling]].

## How it works

- A text stream `W` is derived from a **Whisper** transcript of *Moshi's own* speech (not the
  user's), aligned to the 12.5 Hz frame rate using word-level timestamps.
- Special **`PAD`** / **`EPAD`** tokens fill the gaps between words (≈65% of tokens are padding
  in English); `EPAD` marks where padding ends and the next word begins.
- The text token is placed **first** in each step's stack, so it conditions the semantic then
  acoustic tokens for that step — extending hierarchical semantic→acoustic generation with a
  text prefix.

## Why it's the key lever

- Roughly **triples spoken-QA accuracy** versus audio-only Moshi, at almost no extra inference
  cost (17 tokens/step instead of 16).
- The **text↔audio delay** is a single knob that repurposes the *same* model:
  - text **behind** audio → streaming **ASR** (with word alignment),
  - text **ahead** of audio → streaming **TTS**.
- Gives a handle for control: forcing an `EPAD` token makes Moshi start talking immediately.

## Contrast with TML

[[TML-Interaction-Small]] also couples text and audio, but via **co-training** text/audio in
an [[Encoder-Free Early Fusion|encoder-free]] model rather than an explicit per-step text
prefix. Both aim to keep linguistic quality high in a streaming speech model — an open thread
is which generalizes better.

## Contrast with Voxtral

[[Voxtral]] couples text and audio at the **segment** level instead: its
[[Audio-Text Pretraining Patterns]] interleave whole `(audio, transcript)` segments with
`<repeat>` / `<next>` task tokens. Inner Monologue interleaves at the **token/timestep** level
(a per-frame text prefix in one stream). Same goal — ground audio in the text backbone — at very
different granularities, reflecting that Moshi *generates* aligned speech while Voxtral only
*understands* audio into text.

Source: [[Moshi (Kyutai paper)]].
