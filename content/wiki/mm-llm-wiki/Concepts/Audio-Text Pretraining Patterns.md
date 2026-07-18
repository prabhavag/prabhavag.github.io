---
title: "Audio-Text Pretraining Patterns"
type: concept
tags:
  - concept
  - training
  - speech
  - multimodal
sources: 1
---

The core pretraining design of [[Voxtral]] (Mistral, 2025): how to **introduce speech to a
text LLM** from cheap `(audio, transcript)` data. An audio–text corpus is segmented (via VAD +
diarization; transcripts ASR-**pseudo-labeled** when missing) into pairs
`(A₁,T₁), (A₂,T₂), …, (Aₙ,Tₙ)`, then woven into training samples by **two patterns** (following
Spirit-LM / Zeng et al.):

![[voxtral-fig2-pretrain-patterns.png]]
> *Figure 2 — Pretraining patterns. A segment is split into `(Aₙ,Tₙ)` pairs. **Repetition**:
> audio `Aₙ` is followed by its own transcript `Tₙ`, signaled by `<repeat>`. **Continuation**:
> audio `Aₙ` is followed by the *next* segment's text `Tₙ₊₁`, signaled by `<next>`.*

| Pattern | Sample form | Special token | Teaches |
|---|---|---|---|
| **Audio-to-text repetition** | `(Aₙ, Tₙ)` — audio then *its own* transcript | `<repeat>` | explicit speech→text alignment (mimics ASR) → **transcription** |
| **Cross-modal continuation** | `(A₁, T₂, A₃, T₄, …)` — each audio followed by the *next* segment's text | `<next>` | modality-invariant context modeling (resembles QA / dialogue) → **understanding** |

Because the text following an audio segment is ambiguous (repeat vs continue both valid), the
`<repeat>` / `<next>` tokens disambiguate during training **and steer behavior at inference**.
Each pair is a standalone `<bos>…<eos>` sequence (no prior context). Text-only data is mixed in
to preserve the decoder's text ability.

## Why both — the 50/50 result

The two patterns are **complementary and both necessary** (Voxtral §5.3, Fig 9):

- **Repetition only** → strong ASR, but **~0% on Llama QA** (no understanding).
- **Continuation only** → strong Llama QA, but **~60% WER** on ASR (can't transcribe).
- **Balanced 50/50** → recovers *both* ASR and QA to near single-pattern levels.

So transcription and understanding are **separable capabilities** induced by different data
framings, and a single token (`<repeat>`/`<next>`) lets one model toggle between them.

## Relation to the rest of the wiki

- This is [[Voxtral]]'s answer to the **text↔audio coupling** problem that every speech-LLM
  faces. Compare [[Moshi]]'s **[[Inner Monologue]]** (time-aligned text tokens as a *prefix* to
  audio, in the *same* stream) — a different mechanism for the same goal of grounding audio in
  text. Inner Monologue interleaves at the token/timestep level; Voxtral's patterns interleave at
  the *segment* level with explicit task tokens.
- Both rely on **ASR pseudo-labeling** to turn untranscribed audio into supervision.

Source: [[Voxtral (Mistral paper)]].
