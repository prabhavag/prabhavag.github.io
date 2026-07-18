---
title: "FD-bench"
type: entity
entity_type: benchmark
tags:
  - entity
  - benchmark
  - interactivity
sources: 1
---

# FD-bench

A benchmark suite (one of the few) for measuring **interactivity**, used to evaluate
[[TML-Interaction-Small]]. The model is given prerecorded audio and must respond at the
correct times. Versions seen:

- **V1** — simple turn-taking **latency** (seconds). TML-Interaction-Small: **0.40s** (best
  of the compared models).
- **V1.5** — average interaction **quality** across scenarios (interruption, backchannel,
  talking to others, background speech). TML-Interaction-Small: **77.8** vs ~45–54 for GPT-
  realtime and Gemini baselines.
- **V3** — response quality / pass@1 with **audio + tools**. TML: 82.8 / 68.0 (background
  agent enabled).

## Related benchmarks in the same source

- **Audio MultiChallenge** — intelligence / instruction-following (the "intelligence" axis
  paired against FD-bench's interactivity axis).
- **QIVD** (video+audio QA, streaming), **BigBench Audio**, **IFEval (VoiceBench / text)**,
  **Harmbench** (refusal).
- Internal proactivity benchmarks: **TimeSpeak**, **CueSpeak**, and adapted **RepCount-A**,
  **ProactiveVideoQA**, **Charades** — where no existing model performs meaningfully.

## Example clips (from the source)

The "New dimensions of interactivity" section ships 5 audio-comparison examples, each pitting
the input against our model and the baselines:
`https://thinkingmachines.ai/audio/interaction-models/example-{1..5}/{input,our_model,gpt_realtime_2,gpt_realtime_2_thinking_xhigh,gemini_think_high,gemini_think_minimal}.wav`.
Example-5 also has a benchmark video:

<video width="100%" height="315" controls preload="metadata" src="https://thinkingmachines.ai/audio/interaction-models/example-5/video.mp4"></video>

Full catalog on [[Interaction Models (TML blog)]].

## Caveats

Some baseline numbers are self-reported via Scale AI / Artificial Analysis, not run by the
authors — worth keeping in mind when comparing.

The baseline set is GPT-realtime and Gemini-live; notably absent is [[Moshi]], an open
[[Full-Duplex Spoken Dialogue|full-duplex]] model that would be a natural point of comparison
(open thread — see [[Moshi (Kyutai paper)]]).

Source: [[Interaction Models (TML blog)]].
