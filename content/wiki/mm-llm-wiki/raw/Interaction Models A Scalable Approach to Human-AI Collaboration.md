---
title: "Interaction Models: A Scalable Approach to Human-AI Collaboration"
source: "https://thinkingmachines.ai/blog/interaction-models/#capabilities"
author:
  - "[[Thinking Machines Lab]]"
published: 2026-05-10
created: 2026-06-01
description: "Interaction models move beyond turn-based AI interfaces by handling multimodal, real-time collaboration natively across audio, video, and text."
tags:
  - "clippings"
---
Today, we’re announcing a research preview of interaction models: models that handle interaction natively rather than through external scaffolding. We think interactivity should scale alongside intelligence; the way we work with AI should not be treated as an afterthought. Interaction models let people collaborate with AI the way we naturally collaborate with each other—they continuously take in audio, video, and text, and think, respond, and act in real time.

We train an interaction model from scratch. To ensure real-time responsiveness, we adopt a multi-stream, micro-turn design. Our research preview demonstrates qualitatively new interaction capabilities, as well as state-of-the-art combined performance in intelligence and responsiveness.

<!-- Media restored from source (the web clipper dropped page media). Videos embedded as HTML
     <iframe>s — this vault auto-localizes `![](url)` markdown embeds (breaks YouTube), but
     leaves iframes alone. Source text is unchanged. -->
**Hero video — "Introducing interaction models | Thinking Machines Lab":**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/A12AVongNN4" title="Introducing interaction models" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## The collaboration bottleneck#

AI labs often treat the ability for AI to work autonomously as the model’s most important capability. As a result, today’s models and interfaces aren’t optimized for humans to remain in the loop.

Autonomous interfaces are valuable, but in most real work, users can’t fully specify their requirements upfront and walk away—good results benefit from a collaborative process where the human stays in the loop, clarifying and giving feedback along the way. However, humans increasingly get pushed out not because the work doesn’t need them, but because the interface has no room for them. Instead, people are most effective when they can collaborate with AI the same way we do with other people: messaging, talking, listening, seeing, showing, and interjecting as needed—and for the model to do the same.,

In order to resolve this, we need to move beyond the current turn-based interface for the models. Today’s models experience reality in a single thread. Until the user finishes typing or speaking, the model waits with no perception of what the user is doing or how the user is doing it. Until the model finishes generating, its perception freezes, receiving no new information until it finishes or is interrupted. This creates a narrow channel for human-AI collaboration that limits how much of a person’s knowledge,, intent, and judgement can reach the model, and how much of the model’s work can be understood. Picture trying to resolve a crucial disagreement over email rather than in person.

At Thinking Machines, we believe we can solve this bandwidth bottleneck by making **AI interactive in real time across any modality**. This enables AI interfaces to meet humans where they are, rather than forcing humans to contort themselves to AI interfaces.

Most existing AI models bolt on interactivity with a harness: stitching components together to emulate interruptions, multimodality, or concurrency. However, “the bitter lesson” suggests that these hand-crafted systems will be outpaced by the advance of general capabilities. **For interactivity to scale with intelligence, it must be part of the model itself.** With this approach, scaling a model makes it smarter *and* a better collaborator.

## Capabilities#

Having interactivity be part of the model unlocks a variety of capabilities that would otherwise need to be implemented in the harness.

- **Seamless dialog management.** The model tracks implicitly whether the speaker is thinking, yielding, self-correcting, or inviting a response. There is no separate dialog management component.
- **Verbal and visual interjections.** The model jumps in as needed depending on the context, not only when the user finishes speaking.
- **Simultaneous speech.** The user and the model can speak concurrently (e.g. live translation)
- **Time-awareness.** The model has a direct sense of elapsed time.
- **Simultaneous tools calls, search, and generative UI.** While speaking and listening to the user, the model can concurrently search, browse the web, or generate UI—weaving back results into the conversation as needed.

In a longer real session, all of this happens continuously, creating an experience that feels more like collaborating and less like prompting.

None of the brands or products appearing in these videos are associated with Thinking Machines Labs. These videos are to demonstrate the model's capabilities and do not indicate sponsorship or partnership.

<!-- Capability demo videos restored from source (#capabilities). All YouTube, embedded. -->
**Seamless dialog management:**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/Ys6i_MGnjUA" title="Seamless dialog management" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Verbal interjection:**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/_SsogUUZP2o" title="Verbal interjection" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Visual interjection:**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/n2GXGjy41HQ" title="Visual interjection" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Simultaneous speech:**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/2ky5MXBvZP8" title="Simultaneous speech" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Time awareness:**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/sqJNfze19jA" title="Time awareness" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Simultaneous tool calls and search:**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/ly3GtaiRFyo" title="Simultaneous tool calls and search" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Generative UI:**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/GL1waQJsV9c" title="Generative UI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Longer real session:**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/qXdYDUqxSxA" title="Longer real session" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Continuous collaboration:**
<iframe width="100%" height="315" src="https://www.youtube.com/embed/iVDJ8O89ERg" title="Continuous collaboration" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Our approach#

Time-aligned micro-turn based

Interaction is grounded in time with continuous input and output streams split into micro-turns

<!-- Figure 1 restored as a screenshot of the original (animated SVG/CSS) diagram. -->
![[fig1-timeline.png]]

Turn-based models see an alternating token sequence. Time-aware interaction models see a continuous stream of micro-turns, so silence, overlap, and interruption remain part of the model's context.

An interaction model is in constant two-way exchange with the user—perceiving and responding at the same time. Some domains take such interactivity as a given—the physical world demands that robotics and autonomous vehicles operate in real time. Audio full-duplex models are another example where interaction is bidirectional and continuous.

Applying the same principle, we set out to build an interaction model native to this regime—one that perceives and responds in the same continuous loop, across audio, video, and text. The result is a system architected around two ideas: a time-aware interaction model that maintains real-time presence, and an asynchronous background model that handles sustained reasoning, tool use, and longer-horizon work.

### System overview#

The interaction model is in constant exchange with the user. When a task requires deeper reasoning than can be produced instantaneously, the interaction model delegates to a background model that runs asynchronously. The interaction model remains present throughout — answering follow-ups, taking new input, holding the thread — and integrates background results into the conversation as they arrive.

This split lets the user benefit from both responsiveness as well as the full extent of intelligence: the planning, tool-use, and agentic workflows of reasoning models at the response latency of non-thinking ones. Note that both the background and interaction models are intelligent — on its own, the interaction model is also competitive on both interactive and intelligence benchmarks.

<!-- Figure 2 restored as a screenshot of the original (interactive SVG/CSS) diagram. -->
![[fig2-system-overview.png]]

The user continuously interacts with the interaction model, while the background model performs asynchronous tasks. Both systems share their context.

### The interaction model#

Our starting point is continuous audio and video — modalities that are inherently real-time. Text can wait, but a live conversation cannot. By designing around the hardest case first, we arrive at an architecture that is natively multimodal, time-aware, and capable of handling concurrent input and output streams across all modalities. Several design choices make this possible.

**Time-aligned micro-turns.** The interaction model works with micro-turns continuously interleaving the processing of 200ms worth of input and generation of 200ms worth of output. Rather than consuming a complete user-turn and generating a complete response, both input and output tokens are treated as streams. Working with 200ms chunks of these streams enables near real-time concurrency of multiple input and output modalities.

Human perception

input 0 input 1 input 2 input 3 input 4

output 0 output 1 output 2 output 3

Model token sequence

input 0 output 0 input 1 output 1 input 2 output 2 input 3 output 3 input 4

<!-- Figure 3 restored as a screenshot of the original (animated SVG/CSS) diagram. -->
![[fig3-token-timeline.png]]

Human perception preserves concurrent input and output streams, while the model receives a single interleaved token sequence.

With this design, there are no artificial turn boundaries that the model must adhere to. In contrast, most existing real-time systems require a harness that predicts turn boundaries in order for the turn-based models to feel real-time and responsive. This harness is made out of components like voice-activity-detection (VAD) that are meaningfully less intelligent than the model itself. This precludes a variety of interaction modes like proactive interjections (“interrupt when I say something wrong”) or reactions to visual cues (“tell me when I’ve written a bug in my code”). Moreover, the model can do things like speak while listening (“translate from spanish to english live”) or watching (“live-commentate this sports game”).

Thus, all of these different interaction modes that require special harnesses today become special-cases of what the model can do and improve in quality as we scale up model size and training data.

**Encoder-free early fusion.** Rather than processing audio and video through large, standalone encoders, we opt for a system with minimal pre-processing. Many omnimodal models require training a separate encoder (e.g. Whisper-like) or decoder (e.g. TTS model-like). We instead take in audio signals as dMel ([Bai, et al. 2024](https://arxiv.org/abs/2407.15835)) and transform it via a light-weighted embedding layer. Images are split into 40x40 patches which are encoded by an hMLP ([Touvron et al. 2022](https://arxiv.org/abs/2203.09795)). For the audio decoder we use a flow head ([Lipman at al. 2022](https://arxiv.org/abs/2210.02747)). All components are co-trained from scratch together with the transformer.

<svg viewBox="0 52 620 482" width="620" height="482" role="img" aria-label="Diagram of the interaction model architecture for a single 200ms micro-turn. Text, frame, and audio inputs are embedded into a shared Transformer, which produces text and mel outputs. A 200ms span is shown under the input columns." style="block-size: auto; inline-size: 100%; max-inline-size: 100%; min-inline-size: 0px; min-width: 0px; height: auto !important; max-width: 100% !important; width: 100% !important;"><g><path d="M 82 430 V 422" fill="none" stroke="currentColor"></path><path d="M 196 430 V 422" fill="none" stroke="currentColor"></path><path d="M 310 430 V 422" fill="none" stroke="currentColor"></path><path d="M 82 376 V 368" fill="none" stroke="currentColor"></path><path d="M 196 376 V 368" fill="none" stroke="currentColor"></path><path d="M 310 376 V 368" fill="none" stroke="currentColor"></path><path d="M 82 321 V 291" fill="none" stroke="currentColor"></path><path d="M 78 297 L 82 291 L 86 297" fill="none" stroke="currentColor"></path><path d="M 196 321 V 291" fill="none" stroke="currentColor"></path><path d="M 192 297 L 196 291 L 200 297" fill="none" stroke="currentColor"></path><path d="M 310 321 V 291" fill="none" stroke="currentColor"></path><path d="M 306 297 L 310 291 L 314 297" fill="none" stroke="currentColor"></path><path d="M 424 243 V 213" fill="none" stroke="currentColor"></path><path d="M 538 243 V 213" fill="none" stroke="currentColor"></path><path d="M 424 165 V 149" fill="none" stroke="currentColor"></path><path d="M 420 155 L 424 149 L 428 155" fill="none" stroke="currentColor"></path><path d="M 538 165 V 149" fill="none" stroke="currentColor"></path><path d="M 534 155 L 538 149 L 542 155" fill="none" stroke="currentColor"></path></g><g><g data-ia-modality="text" tabindex="0"><rect x="30" y="433" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="91" y="453">Text</text> </g><g data-ia-modality="image video" tabindex="0"><rect x="144" y="433" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="206" y="453">Frame</text> </g><g data-ia-modality="audio" tabindex="0"><rect x="258" y="433" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="319" y="453">Audio</text> </g></g><g><g data-ia-modality="text" tabindex="0"><rect x="30" y="325" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="82" y="345">Embedding</text> </g><g data-ia-modality="text" tabindex="0"><rect x="30" y="379" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="82" y="399">Tokens</text> </g><g data-ia-modality="image video" tabindex="0"><rect x="144" y="379" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="196" y="399">40x40 Patch</text> </g><g data-ia-modality="image video" tabindex="0"><rect x="144" y="325" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="196" y="345">hMLP</text> </g><g data-ia-modality="audio" tabindex="0"><rect x="258" y="379" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="310" y="399">dMel</text></g><g data-ia-modality="audio" tabindex="0"><rect x="258" y="325" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><foreignObject x="258" y="325" width="104" height="40"><div xmlns="http://www.w3.org/1999/xhtml">Bag of embeddings</div></foreignObject></g></g><rect x="30" y="247" width="560" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="310" y="267" fill="currentColor">Transformer</text> <g><g data-ia-modality="text" tabindex="0"><rect x="372" y="105" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="433" y="125">Text</text> </g><g data-ia-modality="text" tabindex="0"><rect x="372" y="169" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="424" y="189">Unembedding</text> </g><g data-ia-modality="audio" tabindex="0"><rect x="486" y="105" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="545" y="125">Mel</text> </g><g data-ia-modality="audio" tabindex="0"><rect x="486" y="169" width="104" height="40" rx="7" ry="7" fill="none" stroke="currentColor"></rect><text x="538" y="189">Flow</text></g></g></svg>

An illustration of the interaction model architecture for a single 200ms micro-turn. The model takes in any subset of text, audio, or video and predicts text and audio.

**Inference optimization.** At inference time, 200ms chunks require frequent prefills and decodes of small sizes, each having to meet strict latency constraints. Unfortunately, existing LLM inference libraries are not optimized for frequent small prefills—they often have a significant amount of overhead per turn. To address this, we implemented streaming sessions. The client sends each 200ms chunk as a separate request, while the inference server appends these chunks into a persistent sequence in GPU memory. This avoids frequent memory reallocations and metadata computations, and we’ve upstreamed a [version of this feature](https://github.com/sgl-project/sglang/pull/19171) to SGLang. In addition, we also optimized our kernels for latency as well as the shapes we see for bidirectional serving. For example, we use a gather+gemv strategy for MoE kernels instead of the standard grouped gemm, as in prior work from [PyTorch](https://www.thonking.ai/p/short-supporting-mixtral-in-gpt-fast) and [Cursor](https://cursor.com/blog/warp-decode).

**Trainer-sampler alignment.** We’ve found bitwise trainer-sampler alignment to be useful for training stability as well as debugging the various components of our system. We implement [batch-invariant kernels](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) with minimal (<5%) e2e performance overhead. To highlight two particular kernels:

- **All-reduce and reduce-scatter**: We use NVLS to implement low-latency comm kernels which are deterministic on Blackwell, and achieve bitwise alignment between somewhat different parallelism strategies (i.e. [Sequence Parallelism](https://arxiv.org/abs/2205.05198) and Tensor Parallelism).
- **Attention**: The primary challenge with attention is Split-KV, which can typically lead to inconsistent accumulation orders between decode and prefill. However, we can maintain consistent accumulation order by choosing to split consistently between decode and prefill. For example, we could split SMs to process 4096 tokens at a time (left-aligned), achieving good efficiency in both prefill and decode.

**Coordination between interaction and background models.** When the interaction model delegates, it sends a rich context package — not a standalone query, but the full conversation. Results stream back as the background model produces them, and the interaction model interleaves these updates into the conversation at a moment appropriate to what the user is currently doing, rather than as an abrupt context switch.

**Safety.** Because real-time interaction stresses safety differently than turn-based exchanges, our safety work focused on two axes: modality-appropriate refusals and long-horizon robustness. To make refusals colloquial in speech, we use a text-to-speech model to generate refusal and over-refusal training data covering a range of disallowed topics, with the refusal boundary calibrated to favor naturally-phrased, but no less firm, refusals. To improve robustness across extended speech-to-speech conversations, we used an automated red-teaming harness to generate multi-turn refusal data, while maintaining close behavioral parity with the model’s text-based refusals.

## Benchmarks#

### Intelligence and interactivity frontier#

We show that our interaction model, named `TML-Interaction-Small`, is the first model that has both strong intelligence/instruction following **and** interactivity. To measure interaction quality we use FD-bench which is one of the few existing benchmarks intended to measure interactivity. In FD-bench v1.5, the model is given prerecorded audio, and must respond at certain times. This benchmark measures model behavior across several scenarios: user interruption, user backchannel, talking to others, and background speech. Our model scores well in all of these areas. To quantify intelligence we use Audio MultiChallenge, a common benchmark that tracks intelligence and instruction following.

<svg viewBox="0 0 960 535" aria-hidden="true" focusable="false"><text x="502" y="30" style="font-size: 20px;font-size: 32px !important;" fill="currentColor"><tspan x="502">Intelligence (Audio MultiChallenge, APR) <tspan font-style="italic">vs.</tspan> Interaction Quality</tspan> <tspan x="502" dy="36">(FD-bench v1.5, average quality)</tspan> </text><g transform="translate(0 50)"><g><line x1="92" y1="42" x2="92" y2="370"></line><line x1="183.11" y1="42" x2="183.11" y2="370"></line><line x1="274.22" y1="42" x2="274.22" y2="370"></line><line x1="365.33" y1="42" x2="365.33" y2="370"></line><line x1="456.44" y1="42" x2="456.44" y2="370"></line><line x1="547.56" y1="42" x2="547.56" y2="370"></line><line x1="638.67" y1="42" x2="638.67" y2="370"></line><line x1="729.78" y1="42" x2="729.78" y2="370"></line><line x1="820.89" y1="42" x2="820.89" y2="370"></line><line x1="912" y1="42" x2="912" y2="370"></line><line x1="92" y1="42" x2="912" y2="42"></line><line x1="92" y1="51.11" x2="912" y2="51.11"></line><line x1="92" y1="96.67" x2="912" y2="96.67"></line><line x1="92" y1="142.22" x2="912" y2="142.22"></line><line x1="92" y1="187.78" x2="912" y2="187.78"></line><line x1="92" y1="233.33" x2="912" y2="233.33"></line><line x1="92" y1="278.89" x2="912" y2="278.89"></line><line x1="92" y1="324.44" x2="912" y2="324.44"></line><line x1="92" y1="370" x2="912" y2="370"></line></g><g><line x1="92" y1="370" x2="912" y2="370"></line><line x1="92" y1="42" x2="92" y2="370"></line></g><g><text x="92" y="402" style="font-size: 19px;font-size: 20px;">40</text> <text x="183.11" y="402" style="font-size: 19px;font-size: 20px;">45</text> <text x="274.22" y="402" style="font-size: 19px;font-size: 20px;">50</text> <text x="365.33" y="402" style="font-size: 19px;font-size: 20px;">55</text> <text x="456.44" y="402" style="font-size: 19px;font-size: 20px;">60</text> <text x="547.56" y="402" style="font-size: 19px;font-size: 20px;">65</text> <text x="638.67" y="402" style="font-size: 19px;font-size: 20px;">70</text> <text x="729.78" y="402" style="font-size: 19px;font-size: 20px;">75</text> <text x="820.89" y="402" style="font-size: 19px;font-size: 20px;">80</text> <text x="912" y="402" style="font-size: 19px;font-size: 20px;">85</text> </g><g><text x="76" y="376" style="font-size: 19px;font-size: 20px;">20</text> <text x="76" y="284.89" style="font-size: 19px;font-size: 20px;">30</text> <text x="76" y="193.78" style="font-size: 19px;font-size: 20px;">40</text> <text x="76" y="102.67" style="font-size: 19px;font-size: 20px;">50</text> </g><text x="502" y="442" style="font-size: 19px;font-size: 30px !important;" fill="currentColor">Interaction Quality →</text> <text x="-206" y="32" transform="rotate(-90)" style="font-size: 19px;font-size: 30px !important;" fill="currentColor">Intelligence →</text> <g><g transform="translate(784.44 157.16)" data-ii-tooltip-title="TML-interaction-small" data-ii-tooltip-values="Interaction quality 78.0 | Intelligence 43.4"><g><circle r="8.5" fill="#be2121"></circle></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="-18" y="5" text-anchor="end" style="font-size: 18px;font-size: 24px;">TML-small</text> </g><g transform="translate(237.78 110.79)" data-ii-tooltip-title="GPT-realtime-2.0 (xhigh)" data-ii-tooltip-values="Interaction quality 48.0 | Intelligence 48.5"><g><path d="M 0 -10 L 9 8 L -9 8 Z" fill="#5BA3E8"></path></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="18" y="5" text-anchor="start" style="font-size: 18px;font-size: 24px;">GPT-2.0 xhigh</text> </g><g transform="translate(219.56 209.56)" data-ii-tooltip-title="GPT-realtime-2.0 (minimal)" data-ii-tooltip-values="Interaction quality 47.0 | Intelligence 37.6"><g><circle r="8.5" fill="#5BA3E8"></circle></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="18" y="5" text-anchor="start" style="font-size: 18px;font-size: 24px;">GPT-2.0 min</text> </g><g transform="translate(246.89 235.80)" data-ii-tooltip-title="GPT-realtime-1.5" data-ii-tooltip-values="Interaction quality 48.5 | Intelligence 34.7"><g><circle r="8.5" fill="#5BA3E8"></circle></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="0" y="40" text-anchor="middle" style="font-size: 18px;font-size: 24px;">GPT-1.5</text> </g><g transform="translate(183.11 223.68)" data-ii-tooltip-title="Gemini-3.1-flash-live-preview (high)" data-ii-tooltip-values="Interaction quality 45.0 | Intelligence 36.1"><g><path d="M 0 -10 L 9 8 L -9 8 Z" fill="#0F6E56"></path></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="-18" y="5" text-anchor="end" style="font-size: 18px;font-size: 24px;">Gemini high</text> </g><g transform="translate(465.56 308.32)" data-ii-tooltip-title="Gemini-3.1-flash-live-preview (minimal)" data-ii-tooltip-values="Interaction quality 60.5 | Intelligence 26.8"><g><circle r="8.5" fill="#0F6E56"></circle></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="0" y="40" text-anchor="middle" style="font-size: 18px;font-size: 24px;">Gemini min</text> </g></g></g></svg> <svg viewBox="0 0 960 535" aria-hidden="true" focusable="false"><text x="502" y="30" style="font-size: 20px;font-size: 32px !important;" fill="currentColor"><tspan x="502">Intelligence (Audio MultiChallenge, APR) <tspan font-style="italic">vs.</tspan> Responsiveness </tspan><tspan x="502" dy="36">(FD-bench v1, simple turn-taking latency)</tspan> </text><g transform="translate(0 50)"><g><line x1="92" y1="42" x2="92" y2="370"></line><line x1="284.94" y1="42" x2="284.94" y2="370"></line><line x1="526.12" y1="42" x2="526.12" y2="370"></line><line x1="767.29" y1="42" x2="767.29" y2="370"></line><line x1="912" y1="42" x2="912" y2="370"></line><line x1="92" y1="42" x2="912" y2="42"></line><line x1="92" y1="51.11" x2="912" y2="51.11"></line><line x1="92" y1="96.67" x2="912" y2="96.67"></line><line x1="92" y1="142.22" x2="912" y2="142.22"></line><line x1="92" y1="187.78" x2="912" y2="187.78"></line><line x1="92" y1="233.33" x2="912" y2="233.33"></line><line x1="92" y1="278.89" x2="912" y2="278.89"></line><line x1="92" y1="324.44" x2="912" y2="324.44"></line><line x1="92" y1="370" x2="912" y2="370"></line></g><g><line x1="92" y1="370" x2="912" y2="370"></line><line x1="92" y1="42" x2="92" y2="370"></line></g><g><text x="912" y="402" style="font-size: 19px;font-size: 20px;">0.2</text> <text x="767.29" y="402" style="font-size: 19px;font-size: 20px;">0.5</text> <text x="526.12" y="402" style="font-size: 19px;font-size: 20px;">1.0</text> <text x="284.94" y="402" style="font-size: 19px;font-size: 20px;">1.5</text> <text x="92" y="402" style="font-size: 19px;font-size: 20px;">1.9</text> </g><g><text x="76" y="376" style="font-size: 19px;font-size: 20px;">20</text> <text x="76" y="284.89" style="font-size: 19px;font-size: 20px;">30</text> <text x="76" y="193.78" style="font-size: 19px;font-size: 20px;">40</text> <text x="76" y="102.67" style="font-size: 19px;font-size: 20px;">50</text> </g><text x="502" y="442" style="font-size: 19px;font-size: 30px !important;" fill="currentColor">Responsiveness (s) →</text> <text x="-206" y="32" transform="rotate(-90)" style="font-size: 19px;font-size: 30px !important;" fill="currentColor">Intelligence →</text> <g><g transform="translate(815.53 157.16)" data-ii-tooltip-title="TML-interaction-small" data-ii-tooltip-values="Responsiveness 0.40s | Intelligence 43.4"><g><circle r="8.5" fill="#be2121"></circle></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="-18" y="5" text-anchor="end" style="font-size: 18px;font-size: 24px;">TML-small</text> </g><g transform="translate(212.59 110.79)" data-ii-tooltip-title="GPT-realtime-2.0 (xhigh)" data-ii-tooltip-values="Responsiveness 1.65s | Intelligence 48.5"><g><path d="M 0 -10 L 9 8 L -9 8 Z" fill="#5BA3E8"></path></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="18" y="5" text-anchor="start" style="font-size: 18px;font-size: 24px;">GPT-2.0 xhigh</text> </g><g transform="translate(439.29 209.56)" data-ii-tooltip-title="GPT-realtime-2.0 (minimal)" data-ii-tooltip-values="Responsiveness 1.18s | Intelligence 37.6"><g><circle r="8.5" fill="#5BA3E8"></circle></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="18" y="5" text-anchor="start" style="font-size: 18px;font-size: 24px;">GPT-2.0 min</text> </g><g transform="translate(719.06 235.80)" data-ii-tooltip-title="GPT-realtime-1.5" data-ii-tooltip-values="Responsiveness 0.60s | Intelligence 34.7"><g><circle r="8.5" fill="#5BA3E8"></circle></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="0" y="-36" text-anchor="middle" style="font-size: 18px;font-size: 24px;">GPT-1.5</text> </g><g transform="translate(284.94 223.68)" data-ii-tooltip-title="Gemini-3.1-flash-live-preview (high)" data-ii-tooltip-values="Responsiveness 1.50s | Intelligence 36.1"><g><path d="M 0 -10 L 9 8 L -9 8 Z" fill="#0F6E56"></path></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="18" y="5" text-anchor="start" style="font-size: 18px;font-size: 24px;">Gemini high</text> </g><g transform="translate(550.24 308.32)" data-ii-tooltip-title="Gemini-3.1-flash-live-preview (minimal)" data-ii-tooltip-values="Responsiveness 0.95s | Intelligence 26.8"><g><circle r="8.5" fill="#0F6E56"></circle></g><circle r="22" fill="none" stroke="currentColor"></circle><text x="0" y="40" text-anchor="middle" style="font-size: 18px;font-size: 24px;">Gemini min</text></g></g></g></svg>

TML-interaction-small GPT-realtime-2.0 (minimal) GPT-realtime-2.0 (xhigh) GPT-realtime-1.5 Gemini-3.1-flash-live-preview (minimal) Gemini-3.1-flash-live-preview (high)

Intelligence and Interactivity Frontier. Our model dominates interaction quality while being more intelligent than any non thinking model. We achieve the best responsiveness measured as a latency between user and model turns.

For more intelligence, safety, and interactivity/latency results please see the table below. We report our performance on both streaming and turn-based benchmarks.

<table><thead><tr><th colspan="5">Instant</th><th colspan="2">Thinking</th></tr><tr><th>TML-interaction<br>-small</th><th>GPT-realtime-2.0<br>(minimal)</th><th>GPT-realtime-1.5</th><th>Gemini-3.1-flash-live<br>(minimal)</th><th>Qwen 3.5<br>OMNI-plus-realtime</th><th>GPT-realtime-2.0<br>(xhigh)</th><th>Gemini-3.1-flash-live<br>(high)</th></tr></thead><tbody><tr><td rowspan="4"></td><td>FD-bench V1 Turn-taking latency (s) · Audio</td><td>0.40</td><td>1.18</td><td>0.59</td><td>0.57</td><td>2.14</td><td>1.63</td><td>0.94</td></tr><tr><td>FD-bench V1.5 Average · Audio</td><td>77.8</td><td>46.8</td><td>48.3</td><td>54.3</td><td>39.0</td><td>47.8</td><td>45.5</td></tr><tr><td>FD-bench V3 Response Quality (%) /<br>Pass@1 (%) · Audio + Tools</td><td>82.8 <sup>*</sup> / 68.0 <sup>*</sup></td><td>80.0 / 52.0</td><td>77.9 / 55.0</td><td>68.5 / 48.0</td><td>60.0 / 50.0</td><td>81.0 / 58.0</td><td>71.4 / 48.0</td></tr><tr><td>QIVD <sup>**</sup> Accuracy (%) · Video + Audio</td><td>54.0</td><td>57.5</td><td>41.2</td><td>54.7</td><td>59.0</td><td>58.2</td><td>56.1</td></tr><tr><td rowspan="5"></td><td>Audio MultiChallenge APR (%) · Audio</td><td>43.4</td><td>37.6</td><td>34.7</td><td>26.8</td><td>- <sup>***</sup></td><td>48.5</td><td>36.1</td></tr><tr><td>BigBench Audio Accuracy (%) · Audio</td><td>75.7 / 96.5 <sup>*</sup></td><td>71.8</td><td>81.4</td><td>71.3</td><td>73.0</td><td>96.6 <sup>****</sup></td><td>96.6</td></tr><tr><td>IFEval (VoiceBench) Accuracy (%) · Audio</td><td>82.1</td><td>81.7</td><td>68.1</td><td>67.6</td><td>80.3</td><td>83.2</td><td>82.8</td></tr><tr><td>IFEval Accuracy (%) · Text</td><td>89.7</td><td>89.6</td><td>87.5</td><td>85.8</td><td>83.4</td><td>95.2</td><td>90.0</td></tr><tr><td>Harmbench Refusal rate (%) · Text</td><td>99.0</td><td>99.5</td><td>100.0</td><td>99.0</td><td>99.5</td><td>100.0</td><td>98.0</td></tr></tbody></table>

Best per row Best among instant models

\* For benchmarks that require reasoning or tool calls we report our results with background agent enabled.  
\*\* We evaluate Qualcomm IVD in a streaming setting – is a video-audio QA benchmark. In each video clip, somebody performs an action and speaks a question. We evaluate in a streaming setting, sending the raw clip from the beginning and grading the model’s transcript. Following Qwen 3.5 Omni we use a GPT-4o-mini grader.  
\*\*\* Audio MultiChallenge metrics for all the baseline models are reported by Scale AI, where Qwen 3.5 OMNI-plus-realtime is not listed.  
\*\*\*\* Bigbench Audio metrics for all the baseline models are reported by Artificial Analysis, where GPT-realtime-2.0 thinking is on high.

### New dimensions of interactivity#

The existing interactivity-oriented benchmarks above do not adequately capture the qualitative jumps in interaction capabilities we notice. To that end, we have some early work aimed at quantifying these capabilities.

**Time awareness and simultaneous speech.** Turn-based models with a dialog management system do not support accurate time estimation or simultaneous speech. Examples include: “How long did it take me to run one mile?”, “Correct my mispronunciations as you hear them” or “How long did it take me to write this function?"

We created two internal benchmarks to measure these proactive audio capabilities:

- **TimeSpeak:** Tests whether the model can initiate speech at user-specified times while producing the correct content. For example: “I want to practice my breathing, remind me to breathe in and out every 4 seconds until I ask you to stop.”
- **CueSpeak:** Tests whether the model speaks at the appropriate moment with the expected semantically correct response. Dataset entries are created to ensure that the model needs to speak at the same time as the user to get a full score. For example: “Everytime I codeswitch and use another language, give me the correct word in the original language.”

For both benchmarks, each example has a single expected semantic response and timing window. We grade with an LLM judge: A response is counted as correct only if it conveys the expected meaning and is delivered at the appropriate time; failing either criterion receives no credit. We report macro-averaged accuracy across examples.

**Visual proactivity.** Today’s commercial real-time APIs perform turn-detection via audio-only dialogue management harnesses. They respond to spoken turns, but they cannot proactively choose to speak when the visual world changes. For instance, if asked “Please count how many pushups I do” such a system might respond “Sure thing!” and then remain silent – waiting for an audio-only cue that never comes.

We adapted three benchmarks to evaluate visual proactivity of our model:

- **[RepCount-A](https://arxiv.org/abs/2204.01018)** contains videos of repeated actions and is adapted into an online counting task. We stream the video following an audio instruction “Please count out reps for {action}.”. We extract the last number said by the model after the ground truth penultimate rep, and grade by whether it was within one rep of the ground truth. This task measures continuous visual tracking and timely counting.
- **[ProactiveVideoQA](https://arxiv.org/abs/2507.09313)** consists of videos with questions, whose answers become available at specific moments. We stream the question in audio and then the video. We report the paper’s turn-weighted PAUC@ω=0.5 metric (scaled 0-100), averaged across turns and categories. Staying silent scores 25.0.; Higher scores require correct answers at the correct times and incorrect answers are penalized.
- **[Charades](https://arxiv.org/abs/1604.01753)** is a standard temporal action-localization benchmark. Each video contains an action occurring over a labeled time interval. We stream a user audio instruction: “Say ‘start’ when the person starts doing {action} then say ‘Stop’ when they stop.”; then we stream the video. The model is graded by temporal IoU between predicted and the reference intervals.

No existing model can meaningfully perform any of these tasks. For the sake of completeness, we report the results of GPT Realtime-2 (minimal), but all models evaluated perform similar or worse on these tasks, including thinking high models. They stay silent or give incorrect answers.

<video src="https://thinkingmachines.ai/audio/interaction-models/example-5/video.mp4" aria-label="Example 1 input video" controls=""></video>

Examples from our internal audio and video benchmark.

<!-- Audio-comparison examples restored from source: 5 examples, each comparing the input
     against our model and the baselines. Only example-5 has an accompanying video. -->
**Audio benchmark comparison clips** — each example: [input](https://thinkingmachines.ai/audio/interaction-models/example-1/input.wav) vs [our model](https://thinkingmachines.ai/audio/interaction-models/example-1/our_model.wav) vs baselines ([GPT-realtime-2.0](https://thinkingmachines.ai/audio/interaction-models/example-1/gpt_realtime_2.wav), [GPT-realtime-2.0 thinking xhigh](https://thinkingmachines.ai/audio/interaction-models/example-1/gpt_realtime_2_thinking_xhigh.wav), [Gemini think high](https://thinkingmachines.ai/audio/interaction-models/example-1/gemini_think_high.wav), [Gemini think minimal](https://thinkingmachines.ai/audio/interaction-models/example-1/gemini_think_minimal.wav)).

The same six-file set exists for `example-1` … `example-5` — swap the example number in the
path: `https://thinkingmachines.ai/audio/interaction-models/example-{N}/{input,our_model,gpt_realtime_2,gpt_realtime_2_thinking_xhigh,gemini_think_high,gemini_think_minimal}.wav`. Example-5 also has [video.mp4](https://thinkingmachines.ai/audio/interaction-models/example-5/video.mp4) (shown above).

**Future evals.** We believe that interactivity is an important area for future research and we invite the community to contribute benchmarks here. We are launching a research grant to encourage more research into the field of interaction models and human-AI collaboration, including but not limited to new frameworks for assessing interactivity quality, with details coming soon.

## Limitations and future work#

**Long sessions.** Continuous audio and video accumulate context quickly. The streaming-session design handles short and medium interactions well, but very long sessions still require careful context management—an active area of work.

**Compute and deployment.** Streaming audio and video at low latency requires reliable connectivity. Without a good connection, the experience degrades significantly. We believe that this can be improved significantly in the future both by improving system reliability as well as training our model to be more robust to delayed frames.

**Alignment and safety.** A realtime interface opens up an exciting area of research for both alignment and safety. We are collecting feedback and reviewing research grants.

**Scaling model size.** The current `TML-Interaction-Small` is a 276B parameter MoE with 12B active. While we expect the interactivity to improve with model scale, our larger pretrained models are currently too slow to serve in this setting. We plan to release larger models later this year.

**Improved background agents.** Although we have primarily focused on real-time interactivity in this post, agentic intelligence is also an essential capability. In addition to pushing agentic intelligence to the frontier, we believe we have just scratched the surface in how the background agents can work together with the interaction model.

## Tell us what you think, join us#

In the coming months, we will open a limited research preview to collect feedback, with a wider release later this year.

We’d love for you to [join us](https://job-boards.greenhouse.io/thinkingmachines). Please share your thoughts at [interaction@thinkingmachines.ai](mailto:interaction@thinkingmachines.ai).

## Citation#

Please cite this work as:

```
Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration",
Thinking Machines Lab: Connectionism, May 2026.
```

Or use the BibTeX citation:

```
@article{thinkingmachines2026interactionmodels,
  author = {Thinking Machines Lab},
  title = {Interaction Models: A Scalable Approach to Human-AI Collaboration},
  journal = {Thinking Machines Lab: Connectionism},
  year = {2026},
  month = {May},
  note = {https://thinkingmachines.ai/blog/interaction-models/},
  doi = {10.64434/tml.20260511},
}
```

[prev](https://thinkingmachines.ai/blog/on-policy-distillation/ "On-Policy Distillation")