---
title: "Time-Aligned Micro-Turns"
type: concept
tags:
  - concept
  - architecture
  - real-time
sources: 1
---
	
The core mechanism behind [[Interaction Models]]. Rather than processing a whole user turn
then generating a whole response, both input and output are treated as **continuous streams**
split into ~**200ms chunks** that are interleaved into a single token sequence
(`input 0, output 0, input 1, output 1, …`).

### Figure — turn-based vs. time-aligned timeline (source Fig. 1)

![[fig1-timeline.png]]

> *Turn-based models see an alternating token sequence. Time-aware interaction models see a
> continuous stream of micro-turns, so silence, overlap, and interruption remain part of the
> model's context.*

The direct sense of elapsed time is demonstrated in the time-awareness demo:

<iframe width="100%" height="315" src="https://www.youtube.com/embed/sqJNfze19jA" title="Time awareness demonstration" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### Figure — human perception vs. model token sequence (source Fig. 3)

![[fig3-token-timeline.png]]

> *Human perception preserves concurrent input and output streams, while the model receives a
> single interleaved token sequence.*

## Why it matters

- **No artificial turn boundaries** the model must obey — silence, overlap, and interruption
  remain part of context.
- Removes the need for a turn-prediction **harness** (e.g. VAD) that is less intelligent than
  the model.
- Enables modes impossible for turn-based systems: proactive interjections ("interrupt when I
  say something wrong"), speaking while listening (live translation), and reacting to visual
  cues ("tell me when I've written a bug").
- The model gains a direct sense of **elapsed time** (grounding for benchmarks like TimeSpeak
  / CueSpeak).

## Inference cost

200ms chunks mean frequent small prefills/decodes under strict latency budgets — addressed
via *streaming sessions* (chunks appended into a persistent GPU sequence, upstreamed to
SGLang) and latency-tuned kernels. See [[TML-Interaction-Small]].

## Training data

TML doesn't disclose how TML-small is trained on real-world data — see the training-data notes
on [[TML-Interaction-Small]] (undisclosed) and the documented analogue on [[Moshi]].

## Compare: Moshi's multi-stream tokens

[[Moshi]] solves the same "keep silence/overlap/interruption in context" problem in the audio
domain via [[Multi-Stream Audio Modeling]] — two **fixed-rate (12.5 Hz) parallel token
streams** over the [[Mimi (neural audio codec)|Mimi]] codec, rather than interleaved 200 ms
multimodal micro-turns. Same intent ([[Full-Duplex Spoken Dialogue]]), different granularity
and modality coverage.

Source: [[Interaction Models (TML blog)]].
