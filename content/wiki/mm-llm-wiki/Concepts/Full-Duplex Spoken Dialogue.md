---
title: "Full-Duplex Spoken Dialogue"
type: concept
tags:
  - concept
  - speech
  - real-time
  - human-ai-collaboration
sources: 1
---

Spoken interaction in which the system can **listen and speak at the same time** — both parties
hold open channels continuously, as in natural human conversation, rather than alternating in
clean turns. The model "always listens and always generates sound" (speech or silence).

## Why it's hard (and why it matters)

The dominant approach is a **cascade**: voice-activity detection → ASR → text LLM → TTS. This
is *half-duplex* and turn-based, and it fails real conversation in three ways:

- **Latency** compounds across components → multi-second delays (vs ~230 ms human average).
- **Text bottleneck** — paralinguistic cues (emotion, accent, non-speech sounds) are lost.
- **Rigid turns** — cannot represent overlap (10–20% of spoken time), interruptions, or
  backchanneling ("uh-huh", "I see").

Full-duplex models drop the turn boundary entirely and model the conversation as continuous,
overlapping streams.

## Realizations in this wiki

- **[[Moshi]]** ([[Kyutai]]) — speech-to-speech, achieves it via **[[Multi-Stream Audio Modeling]]**
  (parallel user + model token streams) at 160 ms latency. See [[Moshi (Kyutai paper)]].
- **[[Interaction Models]]** ([[Thinking Machines Lab]]) — the same goal generalized to audio +
  video + text, realized via **[[Time-Aligned Micro-Turns]]**. The TML blog cites audio
  full-duplex models (like Moshi) as prior art.

Both reject the "harness of less-intelligent components" (VAD, dialog management, TTS) in
favour of making interactivity native to the model.

## Related

- [[Time-Aligned Micro-Turns]] · [[Multi-Stream Audio Modeling]] · [[Inner Monologue]]
- Measured by interactivity benchmarks like [[FD-bench]].
