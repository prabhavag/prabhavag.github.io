---
title: "Qwen3-Omni Technical Report"
source: "https://arxiv.org/pdf/2509.17765"
author:
  - "[[Qwen Team]]"
published: 2025-09-22
created: 2026-07-11
description: "Technical report for Qwen3-Omni, a natively end-to-end multimodal model (text, image, audio, video) using a Thinker-Talker MoE architecture with the AuT audio encoder, targeting real-time speech interaction while matching same-size unimodal models."
tags:
  - clippings
converted_from: "arXiv LaTeX source (latexpand + pandoc --citeproc)"
---

maketitle thanks aketitle

![[qwen3omni-fig1-capabilities.png]]

> Qwen3-Omni is a unified end-to-end model capable of processing multiple modalities, such as text, audio, image and video, and generating real-time text or speech response. Based on these features, Qwen3-Omni supports a wide range of tasks, including but not limited to voice dialogue, video dialogue, and video reasoning.

# Introduction

Humans perceive visual and auditory inputs in parallel, cognitively process these signals, and emit responses through textual expression, vocalization, and tool-mediated or bodily actions, facilitating information exchange with other organisms and demonstrating intelligence. Building on the rapid advances in the understanding and reasoning capabilities of unimodal large models (Brown et al. 2020; OpenAI 2023; Gemini Team 2024; Anthropic 2023b, 2023a, 2024; Bai, Bai, Chu, et al. 2023; Yang et al. 2024, 2025; Touvron et al. 2023; Dubey et al. 2024; Li et al. 2023; Liu et al. 2023; Zhu et al. 2023; Bai, Bai, Yang, et al. 2023; Bai et al. 2025; Chu et al. 2023, 2024), natively multimodal systems have drawn substantial attention (OpenAI 2024; Comanici et al. 2025; Xu et al. 2025). Human learning typically progresses through the coordinated use of multiple modalities, where complementary specialization and cross-modal synergy improve learning efficiency. However, contemporary LLM-centric multimodal models often exhibit modality trade-offs, with gains in one modality accompanied by degradation in others.

In this report, we take a step toward resolving this limitation by exploring integrated multimodal training within the prevailing LLM-based paradigm. We demonstrate that joint multimodal training can achieve parity across all modalities—i.e., no modality-specific performance degradation—while markedly enhancing cross-modal capabilities such as video understanding. A key ingredient is mixing unimodal and cross-modal data during the early stage of text pretraining. As evidenced by Qwen3-Omni-30B-A3B-Base, its text and vision performance is on par with same-sized single-modal text and vision base models across extensive benchmarks, while simultaneously exhibiting strong audio competence, audiovisual understanding, cross-modal “thinking”, and real-time audiovisual interaction. The development of non-degrading multimodal systems is an achievable objective. Such systems are characterized by two key properties: first, their ability to match the performance of specialized unimodal models in their respective tasks, and second, their capacity to facilitate novel cross-modal reasoning and interaction. These latter capabilities represent a significant advantage, as they are not present in traditional unimodal approaches.

Qwen3-Omni builds on the Thinker–Talker architecture introduced in Qwen2.5-Omni (Xu et al. 2025) and introduces **five key upgrades**: (1) both the Thinker and Talker are upgraded to Mixture-of-Experts (MoE) designs; (2) we replace Whisper audio encoder with our AuT (Audio Transformer) encoder, trained from scratch on 20 million hours of supervised audio, yielding stronger general-purpose audio representations. AuT employs block-wise window attention to enable real-time prefill caching; (3) on the speech generation side, we adopt a multi-codebook representation, whose increased capacity supports faithful modeling of diverse voices, paralinguistic cues, and acoustic phenomena; (4) the Talker shifts from single-track to multi-track codec modeling, autoregressively predicting multiple codebook layers via MTP modules, while the waveform stage (Code2Wav) replaces block-wise DiT with a lightweight convolutional network (ConvNet); and (5) the input and output audio code rates are reduced to 12.5 Hz, with the output codec enabling single-frame, immediate speech synthesis. Taken together, these changes enable low-latency speech interaction under high concurrency in industrial-scale deployments.

Compared with Qwen2.5-Omni, Qwen3-Omni introduces **four major improvements**: (1) support for audio understanding on inputs exceeding 40 minutes; (2) expanded language coverage to 119 written languages, 19 and 10 spoken languages for understanding and generation respectively ; (3) a Thinking model enabling full-modality reasoning, including audio–video and audio-only scenarios; and (4) improved streaming performance with end-to-end latency as low as 234 ms.

Critically, Qwen3-Omni maintains state-of-the-art performance on text and visual modalities without degradation relative to same-size single-model Qwen counterparts. Across 36 audio and audio-visual benchmarks, it achieves open-source SOTA on 32 and sets the SOTA on 22, outperforming strong closed-source systems such as Gemini 2.5 Pro, Seed-ASR, and GPT-4o-Transcribe.

The remainder of this paper is organized as follows. Section 2 presents the algorithms and architecture of Qwen3-Omni. Sections 3 and 4 describe the pretraining and post-training datasets and pipelines, respectively. Section 5 reports the experimental results. Section 6 compares Qwen3-Omni with recent Qwen models of comparable parameter scales, demonstrating multimodal performance without modality-induced degradation.

# Architecture

![[qwen3omni-fig2-thinker-talker.png]]

> The overview of Qwen3-Omni. Qwen3-Omni adopts the Thinker-Talker architecture. Thinker is tasked with text generation while Talker focuses on generating streaming speech tokens by receives high-level representations directly from Thinker. To achieve ultra–low-latency streaming, Talker autoregressively predicts a multi-codebook sequence. At each decoding step, an MTP module outputs the residual codebooks for the current frame, after which the Code2Wav renderer incrementally synthesizes the corresponding waveform, enabling frame-by-frame streaming generation.

## Overview

As shown in Figure 2, Qwen3-Omni employs Thinker-Talker architecture (Xu et al. 2025). Compared with Qwen2.5-Omni, Qwen3-Omni introduces the following changes for greater scalability and control:

- Both the Thinker and Talker adopt Mixture-of-Experts (MoE) architectures to support high concurrency and fast inference.

- Talker no longer consumes the Thinker’s high-level text representations and conditions only on audio and visual multimodal features. This design is motivated by: (i) for textual content, discrete tokens and embeddings are effectively information-equivalent; and (ii) multimodal conditioning is necessary for audio–video–coordinated speech generation such as preserving prosody/timbre in speech translation. Moreover, this decoupling allows external modules (e.g., RAG, function calling, safety filters) to intervene on the Thinker’s textual output and, if desired, supply text to the Talker via controlled preprocessing for streaming synthesis.

- Since textual representations are decoupled, the Thinker and Talker can use distinct system prompts, independently controlling the Thinker’s response style and the Talker’s audio style.

- The Talker adopts a multi-codebook autoregressive scheme: Talker generates one codec frame per step, while the MTP module produces the remaining residual codebooks.

- The Code2Wav is implemented as a lightweight causal ConvNet, simplifying the final stage of audio synthesis.

During training and inference, the Talker directly ingests high-dimensional multimodal features from the Thinker and shares access to the full conversational history. As a result, the system operates as a cohesive single model, enabling end-to-end training and unified inference.

In the following sections, we first introduce with our newly proposed AuT encoder, including its training methodology. Then, describe how Thinker processes various inputs. We then detail Talker’s multi-codebook streaming speech generation. Finally, we highlight a series of improvements on both the understanding and generation modules aimed at achieving ultra–low-latency, end-to-end streaming audio inference.

## Audio Transformer (AuT)

![[qwen3omni-fig3-aut-encoder.png]]

> The overview of AuT. AuT is an attention-encoder-decoder based auto-regressive model, which is trained from scratch on 20 million hours of supervised audio. Qwen3-Omni employs the AuT encoder as the audio encoder to obtain general purpose audio representations at a token rate of 12.5Hz.

Audio Transformer (AuT) is an attention-encoder-decoder model, as is shown in Figure 3, trained from scratch on 20 million hours of supervised audio data. During training, the filter bank features of the audio are downsampled 8 times using Conv2D blocks before the attention layers, reducing the token rate to 12.5 Hz. To learn stronger and more general-purpose audio representations, AuT is trained on large-scale audio datasets with both speech recognition and audio understanding tasks. Specifically, the training data includes 80% Chinese and English pseudo-labeled ASR data, 10% ASR data from other languages, and 10% audio understanding data. To balance the efficiency of real-time prefill caching with the performance for offline audio tasks, AuT utilizes flash attention with dynamic attention window sizes, covering attention query patterns ranging from 1 to 8 seconds. In Qwen3-Omni, we employ the AuT encoder as the audio encoder, which contains approximately 0.6B parameters.

## Perceivation

#### Text, Audio, Image and Video (w/o Audio).

Thinker converts text, audio, image, and video (without audio) into a series of representations for input. For text inputs, we use Qwen’s tokenizer (Yang et al. 2025), which applies byte-level byte-pair encoding with a vocabulary of 151,643 regular tokens. For audio inputs and audio extracted from video, we resample to 16 kHz and convert the raw waveform into a 128 channel mel-spectrogram with a 25 ms window and a 10 ms hop. We adopt AuT encoder as our audio encoder, which is trained from scratch on 20 millions hours of audio data, and each frame of the audio representation corresponds to approximately an 80 ms segment of the original audio signal. Furthermore, we employ the vision encoder from Qwen3-VL, initialized from SigLIP2-So400m (Tschannen et al. 2025) with approximately 543 million parameters, enabling handling of both image and video inputs. The vision encoder is trained on a mixture of image and video data, ensuring strong image understanding and video comprehension. To preserve video information as completely as possible while aligning with the audio sampling rate, we sample video frames at a dynamic frame rate.

#### Video and Multimodal Position Embedding (TM-RoPE)

Drawing inspiration from Qwen2.5-Omni, we employs a Time-aligned Multimodal Rotary Position Embedding (TM-RoPE), which extends the Multimodal Rotary Position Embedding (M-RoPE) (Bai, Bai, Yang, et al. 2023) by incorporating absolute temporal information. TM-RoPE factorizes the conventional rotary position embedding into three distinct dimensions: temporal, height, and width. In the original M-RoPE formulation, temporal dependencies are modeled using the initial 16 rotary angles, which correspond to higher frequencies and exhibit stronger oscillatory patterns. While this design is effective for capturing fine-grained local temporal variations, it can impede the model’s ability to extrapolate over extended sequences. To address this limitation, we introduce a modified allocation of rotary angles. Specifically, the temporal, height, and width dimensions are interleaved and assigned 24, 20, and 20 rotary angles, respectively. This redistribution fosters a more balanced representation of both local semantics and long-range dependencies, thereby enhancing the model’s overall performance. The application of TM-RoPE is tailored to the specific modality of the input data. For text inputs, the three components share identical position identifiers, rendering TM-RoPE functionally equivalent to a one-dimensional RoPE (Su et al. 2024). Similarly, audio inputs utilize shared position IDs but are further augmented with absolute temporal encodings, where each temporal ID corresponds to a duration of 80 ms. For image data, a constant temporal ID is assigned to all visual tokens, while their distinct row and column positions determine the height and width IDs.

In the context of multimodal audiovisual streams, the audio component is encoded with a temporal ID for every 80 ms. The video is treated as a sequence of frames with monotonically increasing temporal IDs that are dynamically adjusted based on their actual timestamps to ensure a consistent temporal resolution of 80 ms per ID. The height and width IDs for video frames are assigned in the same manner as for still images. To prevent positional conflicts when processing multiple modalities, the position numbering is made contiguous, with each subsequent modality commencing from one plus the maximum position ID of the preceding modality. This refined approach to positional encoding enables the model to effectively integrate and jointly model information from diverse modalities. In a departure from Qwen2.5-Omni, which segments audiovisual representations into fixed 2-second chunks, Qwen3-Omni directly aligns these representations using their temporal IDs, which are explicitly anchored to absolute time. This design choice affords the model the flexibility to support streaming inputs of arbitrary duration.

## Speech Generation

For speech synthesis in multi-turn dialogues, our Talker module is conditioned on a rich context inherited from a "Thinker" component, comprising historical textual tokens, multimodal representations, and the current turn’s streamed text. This reliance on long-context information is critical, as high-fidelity speech synthesis must adapt acoustic attributes like prosody, loudness, and emotion to the ongoing discourse, a principle well-established in context-aware generative models.

Architecturally, our approach departs from Xu et al. (2025) by operating directly on RVQ tokens. The Talker employs a hierarchical prediction scheme: the backbone ingests the aggregated codebook features of the current frame and uses a linear head to predict the zeroth codebook, after which a multi-token prediction (MTP) module generates all residual codebooks. This strategy enables the model to learn a complete representation of acoustic details, enhancing vocal expressivity. Consequently, waveform reconstruction is simplified to a lightweight causal ConvNet (Code2Wav), which significantly reduces inference latency and computational cost (FLOPs) while achieving superior audio fidelity compared to more complex DiT-based vocoders.

## Designs for Streaming and Concurrency

In streaming audiovisual interaction scenarios, the first-packet latency is a critical factor affecting user experience, and the model’s concurrency capability is key to reducing service costs and improving response speed. This section discusses how Qwen3-Omni enhances concurrency and reduces first-packet latency through algorithmic and architectural optimizations.

<table>
<caption><strong>The architectural design of Qwen3-Omni-30B-A3B and the end-to-end first-packet latency for Audio/Video (ms).</strong></caption>
<thead>
<tr>
<th style="text-align: left;"><strong>Module</strong></th>
<th style="text-align: center;"><strong>Architecture</strong></th>
<th style="text-align: center;"><strong>Params</strong></th>
<th style="text-align: center;"><strong>Streaming</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Audio Encoder</td>
<td style="text-align: center;">AuT</td>
<td style="text-align: center;">650M</td>
<td style="text-align: center;">✓</td>
</tr>
<tr>
<td style="text-align: left;">Vision Encoder</td>
<td style="text-align: center;">SigLIP2-So400M</td>
<td style="text-align: center;">540M</td>
<td style="text-align: center;">-</td>
</tr>
<tr>
<td style="text-align: left;">Thinker</td>
<td style="text-align: center;">MoE Transformer</td>
<td style="text-align: center;">30B-A3B</td>
<td style="text-align: center;">✓</td>
</tr>
<tr>
<td style="text-align: left;">Talker</td>
<td style="text-align: center;">MoE Transformer</td>
<td style="text-align: center;">3B-A0.3B</td>
<td style="text-align: center;">✓</td>
</tr>
<tr>
<td style="text-align: left;">MTP</td>
<td style="text-align: center;">Dense Transformer</td>
<td style="text-align: center;">80M</td>
<td style="text-align: center;">✓</td>
</tr>
<tr>
<td style="text-align: left;">Code2wav</td>
<td style="text-align: center;">ConvNet</td>
<td style="text-align: center;">200M</td>
<td style="text-align: center;">✓</td>
</tr>
<tr>
<td colspan="4" style="text-align: center;">End-to-End First-Packet Latency: <strong>234/547ms</strong></td>
</tr>
</tbody>
</table>

#### Chunked Prefilling and MoE Architecture.

In Qwen3-Omni, we retain the chunked-prefilling mechanism as implemented in Qwen2.5-Omni, whose audio and vision encoders are capable of outputting chunks along the temporal dimension. During real-time interaction, Thinker and Talker modules perform asynchronous prefilling: when Thinker completes prefilling the current chunk, its output high-level representations are immediately used to prefill the Talker’s current chunk asynchronously, while Thinker prefills its next chunk. This approach significantly reduces the Time-To-First-Token (TTFT) for both the Thinker and the Talker. Architecturally, both Thinker and the Talker in Qwen3-Omni adopt the MoE design, which is highly effective for improving service throughput. Compared to dense models, the MoE architecture significantly decreases IO consumption arising from KV cache during processing of long sequences, thereby increasing tokens per second (TPS) during generation and enhancing concurrency.

#### Streaming Multi-Codebook Codec Generation.

To minimize the user’s waiting time for receiving the first generated packet, we propose a *left context only multi-codebook generation* mechanism. As shown in Figure 2, once Talker generates the first token, the MTP module predicts the remaining tokens for the current frame. These tokens are then decoded into waveform by a streaming multi-codebook codec decoder that only attends to the left context. Unlike Qwen2.5-Omni that requires waiting for sufficient block-context from the Talker before synthesis, Qwen3-Omni can output the waveform immediately after the Talker generates each token, significantly reducing first-packet latency.

#### Lightweight MTP module and ConvNet.

Both the MTP module and codec decoder are lightweight modules, which have low computational FLOPs and support batched inference, making them well-suited for high-concurrency scenarios. The MTP Module is an ultra-lightweight fixed-step autoregressive dense transformer, with low memory bandwidth requirements on inference hardware, thereby naturally enabling efficient batch processing of high throughput requests. Its fixed-step autoregressive inference mechanism allows it to effectively leverage a fixed KV cache memory space for acceleration, achieving low inference latency. Meanwhile, the ConvNet-based codec decoder also achieves high throughput with low latency because its convolutional architecture enjoys extensive hardware acceleration support across diverse inference platforms, and it enables efficient batched inference.

<table id="tab:inference-lantency">
<caption><strong>Theoretical First-Packet Latency of Qwen3-Omni wit Different Concurrency.</strong></caption>
<thead>
<tr>
<th style="text-align: left;"></th>
<th colspan="3" style="text-align: center;"><strong>Qwen3-Omni-30B-A3B</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">2-4</td>
<td style="text-align: center;"><strong>1 Concurrency</strong></td>
<td style="text-align: center;"><strong>4 Concurrency</strong></td>
<td style="text-align: center;"><strong>6 Concurrency</strong></td>
</tr>
<tr>
<td style="text-align: left;">Thinker-Talker Tail Packet Preprocessing Latency</td>
<td style="text-align: center;">72/160ms</td>
<td style="text-align: center;">94/180ms</td>
<td style="text-align: center;">100/200ms</td>
</tr>
<tr>
<td style="text-align: left;">Thinker Time-to-First-Token (TTPT)</td>
<td style="text-align: center;">88/160ms</td>
<td style="text-align: center;">468/866ms</td>
<td style="text-align: center;">673/1330ms</td>
</tr>
<tr>
<td style="text-align: left;">Talker Time-to-First-Token (TTPT)</td>
<td style="text-align: center;">57/210ms</td>
<td style="text-align: center;">145/450ms</td>
<td style="text-align: center;">376/734ms</td>
</tr>
<tr>
<td style="text-align: left;">MTP Module Time Cost Per Token</td>
<td style="text-align: center;">14ms</td>
<td style="text-align: center;">16ms</td>
<td style="text-align: center;">18ms</td>
</tr>
<tr>
<td style="text-align: left;">Codec Decoder Time Cost Per Code</td>
<td style="text-align: center;">3ms</td>
<td style="text-align: center;">5ms</td>
<td style="text-align: center;">5ms</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Overral Latency (Audio/Video)</strong></td>
<td style="text-align: center;"><strong>234/547ms</strong></td>
<td style="text-align: center;"><strong>728/1517ms</strong></td>
<td style="text-align: center;"><strong>1172/2284ms</strong></td>
</tr>
<tr>
<td style="text-align: left;">Thinker Token Generation Rate (TPS)</td>
<td style="text-align: center;">75 tokens/s</td>
<td style="text-align: center;">63 tokens/s</td>
<td style="text-align: center;">53 tokens/s</td>
</tr>
<tr>
<td style="text-align: left;">Talker Token Generation Rate (TPS)</td>
<td style="text-align: center;">140 tokens/s</td>
<td style="text-align: center;">125 tokens/s</td>
<td style="text-align: center;">110 tokens/s</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Generation RTF(Real Time Factor)</strong></td>
<td style="text-align: center;"><strong>0.47</strong></td>
<td style="text-align: center;"><strong>0.56</strong></td>
<td style="text-align: center;"><strong>0.66</strong></td>
</tr>
</tbody>
</table>

Table 1 presents the theoretical first-packet latency for Qwen3-Omni under typical computational resources across varying concurrency scenarios. Experiments are conducted on the vLLM framework (Kwon et al. 2023) to process concurrent audiovisual streams, with optimizations applied via *torch.compile* and CUDA Graph acceleration to the MTP Module and codec decoder. Several factors influence the total first-packet latency. First, the model sizes of Thinker and Talker impact their tail packet preprocessing latency (multi-modal data preprocessing and inference for Audio and Vision Encoder) and Time-To-First-Token (TTPT). Second, the architectures and sizes of the MTP Module and Codec Decoder affect their inference latency. Due to the sequential dependency between these components, the total first-packet latency represents the sum of these individual latencies. As shown in the results, the MoE architecture of Thinker and Talker ensures that their prefill latency and TTPT remain largely unaffected under high concurrency. Meanwhile, the lightweight design of the MTP Module and Codec Decoder minimizes their computational overhead, resulting in a lower impact on first-packet latency. Furthermore, after the initial packet is output and the model starts streaming audio synthesis, the 12.5Hz token rate Talker requires only one token to synthesize 80ms audio. Consequently, the Generation Real Time Factor (RTF) is calculated by dividing the sum of: (1) the time taken by Thinker and Talker to generate one token; and (2) the processing time per token for the MTP Module and Codec Decoder by 80ms. As demonstrated, the RTF consistently remains below 1 across varying concurrency levels, ensuring that users receive continuously streaming audio responses.

# Pretraining

| Modality | \# Langs | Languages |  |
|:---|:--:|:--:|:--:|
| Text | 119 | See Qwen3 for the full list. |  |
| Speech Input | 19 | ar, de, en, es, fr, id, it, ja, ko, ms, nl, pt, ru, th, tr, ur, vi, yue, zh |  |
| Speech Output | 10 | de, en, es, fr, it, ja, ko, pt, ru, zh |  |

**Languages and dialects support of Qwen3-Omni-30B-A3B.**

 Qwen3-Omni is pre-trained on a diverse dataset that encompasses multiple languages and dialects as shown in Table [table:languages] and modalities, including image-text, video-text, audio-text, video-audio, video-audio-text, and pure text corpora. Unlike Qwen2.5-Omni, which uses a single prompt for each task, we employ a wider range of natural language prompts to enhance both the generalization ability and instruction-following capabilities. To achieve robust performance across all modalities, our training strategy incorporates both unimodal and cross-modal data from the early pretraining stage.

The pre-training of Qwen3-Omni is structured into three distinct stages. In the first stage, we lock the LLM parameters and focus on training the vision and audio encoders, utilizing a vast corpus of audio-text and image-text pairs to enhance semantic understanding within the LLM. In the second stage, we unfreeze all parameters and train with a wider range of multimodal data for more comprehensive learning. In the final stage, we use data with a sequence length of 32,768 to enhance the model’s ability to understand complex long-sequence data:

1.  **Encoder Alignment Stage (S1)**: During the initial pretraining phase, the LLM component of Qwen3-Omni is initialized with parameters from Qwen3 (Yang et al. 2025), while the vision encoder is adopted from Qwen3-VL, and the audio encoder is initialized with AuT. The two encoders are trained separately on the fixed LLM, with both initially focusing on training their respective adapters before training the encoders. We abandon the stage used in (Bai et al. 2025; Xu et al. 2025) where the encoder and adapter are trained jointly while keeping the LLM frozen, because this approach may cause the encoder to compensate for the limitations of the frozen LLM, which can lead to degraded perception capabilities.

2.  **General Stage (S2)**: The second phase of pretraining utilizes a large-scale dataset containing approximately 2 trillion tokens, with the following distribution across modalities: text (0.57 trillion), audio (0.77 trillion), image (0.82 trillion), video (0.05 trillion), and video-audio (0.05 trillion). During this stage, the introduction of more diverse multimodal data and tasks enhances the model’s understanding and interaction capabilities in auditory, visual, textual, and audiovisual information.

3.  **Long Context Stage (S3)**: In the final pre-training phase, we increased the maximum token length from 8,192 to 32,768 and also raised the proportion of long audio and long video in the training data. Experimental results indicate that these adjustments lead to significant improvements in the model’s ability to understand long sequence data.

# Post-training

## Thinker

The post-training phase comprises a three-stage training process for Thinker, enabling Qwen3-Omni to possess instruction-following capabilities. The dataset, designed in the ChatML (OpenAI 2022) format, includes pure text-based dialogue data, visual modality conversation data, audio modality conversation data, and mixed-modality conversation data.

In the first stage, we introduce a lightweight Supervised Fine-Tuning (SFT) to bridge the gap between pretrained representations and downstream task requirements through targeted instruction optimization. SFT deliberately diverges from the pretraining data schema while maintaining architectural consistency with the pretrained model, enabling efficient knowledge transfer and preserving the completeness of the pretrained features.

The second stage adopts the Strong-to-Weak Distillation pipeline as described in Qwen3 (Yang et al. 2025) to further improve model performance. This distillation process consists of two main phases:

1.  **Off-policy Distillation**: In the initial phase, outputs generated by teacher models are combined to provide response distillation. This helps lightweight student models acquire fundamental reasoning abilities, establishing a strong foundation for subsequent on-policy training.

2.  **On-policy Distillation**: In the second phase, the student model generates the responses based on sampled prompts. These on-policy sequences are then used for fine-tuning, where the student’s predicted logits are aligned with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) by minimizing the KL divergence.

Finally, we leverage GSPO (Zheng et al. 2025) to comprehensively enhance the model’s capabilities and stability across various modalities, including text, image, video, and audio. To provide feedback for the aforementioned modalities, we employ two different types of rewards:

- **Rule-based Reward**: For verifiable multimodal tasks (e.g., mathematics, coding, instruction following), the reward signal is derived from a set of predefined rules. Well-designed rule-based rewards can assess the correctness of model outputs with high precision, preventing issues like reward hacking.

- **Model-based Reward**: To assess performance on multimodal tasks that lack objective, predefined evaluation metrics, we adopt an LLM-as-a-judge protocol. The role of the automated evaluator is filled by Qwen3 for general tasks, while the specialized vision-language model, Qwen2.5-VL, is used for visually-grounded tasks. To ensure a more robust and grounded assessment, the LLM evaluator is furnished with the corresponding ground-truth or reference answer for a given query, where applicable.

## Talker

We introduce a four-stage training process for Talker, enabling Qwen3-Omni to generate speech response simultaneously with text. All training data is structured in the ChatML format to ensure consistency with Thinker.

In the first stage, we leverage hundreds of millions of speech data with multimodal context to train Talker, establishing a monotonic mapping from multimodal representation to speech. In the second stage, we perform Continual Pretraining (CPT) with high-quality data, which alleviates hallucinations caused by noisy data in the first stage and significantly improve the quality of generated speech. Concurrently, we perform long-context training that enhances Talker’s ability to process extended and complex inputs and generate contextually appropriate speech response. In the third stage, to improve the generalization of multilingual speech generation and system stability, we construct preference pairs from diverse multilingual speech samples and optimize the model using Direct Preference Optimization (DPO) (Rafailov et al. 2023). Finally, we apply speaker fine-tuning on the aforementioned base model, enabling Talker to adopt specific voices while refining the naturalness, expressiveness, and controllability of its speech response.

## Captioner

Captioning is a foundational task in multimodal understanding, integral to the training and evaluation of large multimodal models. However, the vast majority of existing research has concentrated on visual captioning, largely neglecting the audio modality. This omission is significant, as auditory perception is a crucial component of human sensory experience and interaction with the world. To address this gap and facilitate more comprehensive research in multimodal perception, we introduce the Qwen3-Omni-30B-A3B-Captioner. This model was developed by fine-tuning the Qwen3-Omni-30B-A3B on a large-scale dataset of detailed audio descriptions. The resulting system generates detailed, low-hallucination captions for arbitrary audio inputs. The **Appendix** 9.2 provides qualitative results that demonstrate our model’s captioning capabilities across diverse acoustic scenarios.

# Evaluation

A comprehensive evaluation was performed on a suite of models, including Qwen3-Omni-30B-A3B-Instruct, Qwen3-Omni-30B-A3B-Thinking, and two in-house developed variants, designated Qwen3-Omni-Flash-Instruct and Qwen3-Omni-Flash-Thinking. These “Flash” models were designed to improve both computational efficiency and performance efficacy, integrating new functionalities, notably the support for various dialects. The evaluation results are divided into two main categories: understanding (X$`\to`$Text) and speech generation (X$`\to`$Speech).

## Evaluation of X$`\to`$Text

In this section, we evaluate Qwen3-Omni’s ability to comprehend various multimodal inputs (text, audio, vision, and audiovisual video) and generate textual responses.

#### Text$`\to`$Text

Our evaluation of Qwen3-Omni on text $`\to`$ text primarily focuses on general tasks, reasoning ability, coding ability, alignment tasks, agent, and multilingual tasks. Specifically, we utilize MMLU-Redux (Gema et al. 2024) and GPQA (Rein et al. 2023) for general tasks, AIME25 (AIME 2025) and ZebraLogic (Lin et al. 2025) for reasoning evaluation, MultiPL-E (Cassano et al. 2023) for coding, IFEval (Zhou et al. 2023), Creative Writing V3 (Paech 2024) and WritingBench (Wu et al. 2025) for alignment tasks, BFCL-v3 (Yan et al. 2024) for agent evaluation, MultiIF (He et al. 2024) and PolyMath (Y. Wang et al. 2025) for multilingual tasks.

#### Audio$`\to`$Text

The evaluation can be categorized into basic audio tasks, including Automatic Speech Recognition (ASR), Speech-to-Text (S2TT), and Music Understanding, as well as advanced audio tasks, including Voice Chatting and Audio Reasoning. For music understanding, we use RUL-MuchoMusic (Zang et al. 2025) for a comprehensive evaluation of the music understanding capabilities of the model. We utilize MMAU (Sakshi et al. 2024) and MMSU (D. Wang et al. 2025) for audio reasoning tasks, VoiceBench (Yiming Chen et al. 2024) for voice-chatting tasks. We also employ multiple datasets including GTZAN (Tzanetakis and Cook 2002), four subsets of MTG-Jamendo (MTG, (Bogdanov et al. 2019)), and MagnaTagATune (Law et al. 2009) to evaluate the model’s capabilities across various music information retrieval tasks including genre identification, emotion and theme recognition, instrument recognition and music keyword annotation. We follow the evaluation set composition in MARBLE (Yuan et al. 2023) for GTZAN, MTG-Jamendo and MagnaTagATune.

#### Vision$`\to`$Text

The evaluation of the model’s vision-to-text capabilities encompasses a suite of benchmarks targeting diverse and challenging tasks. To assess performance in general visual question answering, the model is evaluated on MMStar (L. Chen et al. 2024), HallusionBench (Guan et al. 2024), and MM-MT-Bench (Agrawal et al. 2024). For the specialized domain of mathematical and STEM reasoning, we utilize MathVista (Lu et al. 2024), MathVision (K. Wang et al. 2024), MMMU (Yue et al. 2023), and MMMU-Pro (Yue et al. 2024). The model’s proficiency in document understanding is measured using the AI2D (Kembhavi et al. 2016) and ChartQA (Masry et al. 2022) benchmarks. Furthermore, the model’s numerical reasoning and counting abilities are specifically tested on CountBench (Paiss et al. 2023). To evaluate performance on dynamic visual data, we report results on three long video understanding benchmarks: Video-MME (Fu et al. 2024), LVBench (W. Wang et al. 2024), and MLVU (J. Zhou et al. 2025).

#### AudioVisual Video$`\to`$Text

To evaluate the model’s ability to process dynamic multi-modal information, we first assessed its performance on the WorldSense benchmark (Hong et al. 2025). This benchmark is designed to measure the integration of visual and auditory signals, a foundational capability for operating in complex, open-world environments. To further examine the model’s higher-order cognitive functions, we then evaluated its performance on two audiovisual reasoning benchmarks: DailyOmni (Z. Zhou et al. 2025) and VideoHolmes (Cheng et al. 2025).

### Performance of Text$`\to`$Text

We compare Qwen3-Omni with other leading large language models (thinking or instruct). According to Table Table 4 and Table 5, notably, despite a smaller parameter count, Qwen3-Omni-30B-A3B-Instruct surpasses the performance of the larger open-source model Qwen3-235B-A22B Non-Thinking and the formidable closed-source model GPT-4o-0327 across a suite of benchmarks, including GPQA, AIME25, ZebraLogic, WritingBench, and PolyMath. Concurrently, Qwen3-Omni-30B-A3B-Thinking demonstrates performance comparable to that of Gemini-2.5-Flash-Thinking and Qwen3-235B-A22B Non-Thinking. Furthermore, Qwen3-Omni-30B-A3B exhibits textual capabilities on par with its text-only counterparts, namely the Qwen3-30B-A3B-Instruct-2507 and Qwen3-30B-A3B-Thinking-2507.

### Performance of Audio$`\to`$Text

We compare Qwen3-Omni with other leading specialist and generalist models on ASR & S2TT, voice-chatting, audio reasoning, and music understanding benchmarks. For brevity, we defer the results of the Qwen3-Omni-Thinking model on ASR & S2TT and music understanding to the **Appendix** 9.1.

As shown in Table Table 6, Qwen3-Omni-Instruct achieves state-of-the-art En & Zh ASR and lyric ASR performance on Librispeech, Wenetspeech, Fleurs, CommonVoice, Opencpop-test and MIR-1K (vocal). It also delivers better or comparable performance with other specialist or generalist models like Voxtral-Small and Gemini-2.5-Pro on Multilingual ASR and S2TT. These results show a strong performance of Qwen3-Omni in speech recognition and speech translation.

Additionally, on VoiceBench shown in Table Table 7, Qwen3-Omni-Thinking achieves an impressive average score of 89.5, surpassing all other audio language models except Gemini-2.5-Pro (89.6). This showcases our model’s strong capabilities in speech interaction. Qwen3-Omni also demonstrates impressive performance in audio reasoning, outperforming the powerful closed-source models Gemini-2.5-Pro and Gemini-2.5-Flash on the MMAU benchmark, as well as Gemini-2.5-Flash and GPT-4o-Audio on MMSU. These results demonstrate the powerful capabilities of Qwen3-Omni in general audio understanding and reasoning.

For music understanding, we compare Qwen3-Omni-Instruct with both generalist audio language models and specialist models in Table Table 8. For multi-label classification tasks on MTG-Jamendo and MagnaTagATune, we use micro F1 to compare with BERT-like music specialists instead of AP/AUROC, as language models output discrete label sets without calibrated per-label probabilities/scores required by ranking-based metrics. It is shown in Table Table 8 that Qwen3-Omni-Instruct achieve state-of-the-art performance on RUL-MuchoMusic. On GTZAN, MTG-Jamendo, and MagnaTagATune, the scores of Qwen3-Omni-Instruct also significantly surpass other audio language models, including Gemini-2.5-Pro and GPT-4o-Audio, as well as self-supervised music specialist models probed on the respective datasets. These results demonstrate the superior capabilities of Qwen3-Omni-Instruct across a variety of music understanding tasks.

### Performance of Vision $`\to`$ Text

To comprehensively evaluate the capabilities on Vision $`\to`$ Text, we compare Qwen3-Omni-Instruct with the Qwen2.5-VL-72B and other good-performing closed-source vision-language models. As illustrated in Table Table 9, Qwen3-Omni-Instruct demonstrates comparable performance to Qwen2.5-VL-72B, and attains better results on Math & STEM related tasks like MMMU-Pro<sub>overall</sub>, MathVista<sub>mini</sub>, and MATH-Vision<sub>full</sub>, than other vision language models including GPT4-o and Gemini-2.0-Flash. These results reveal the excellent capability of our model on image understanding and reasoning tasks.

To assess its capabilities, we evaluated the performance of Qwen3-Omni-Thinking against several state-of-the-art reasoning models. The comparative results, summarized in Table Table 10, indicate that our proposed model achieves significant advancements. For instance, on Math and STEM benchmarks, it outperforms the Qwen3-Omni-Instruct baseline by 4.4 points. It is also noteworthy that our Qwen3-Omni-30B-A3B-Thinking model attains a performance level on par with substantially larger baselines, which highlights its excellent balance of effectiveness and computational efficiency. A limitation of the current model is its suboptimal performance on long video benchmarks. This deficiency stems from two architectural constraints: a limited capacity for positional extrapolation and a restricted context length. Addressing these constraints is a key objective for future work.

### Performance of AudioVisual Video$`\to`$Text

As is shown in Table Table 11, the experimental results validate the efficacy of Qwen3-Omni across diverse audiovisual tasks. For general understanding, Qwen3-Omni-Instruct achieves state-of-the-art performance on the WorldSense benchmark, surpassing other Omni models by a substantial margin. This outcome demonstrates its effectiveness in foundational multimodal integration. Moreover, the model exhibits enhanced performance on complex reasoning tasks, as illustrated in Table Table 12, particularly on benchmarks that necessitate reasoning over interconnected audio and visual information. These findings collectively suggest that Qwen3-Omni possesses considerable potential for advanced perception and reasoning in real-world contexts.

## Evaluation of X$`\to`$Speech

In this section, we evaluate the speech generation capabilities of Qwen3-Omni. Due to the lack of relevant assessments, the evaluation of speech generation focuses primarily speech generation given texts, similarity to text-to-speech (TTS), on following three aspects:

- **Zero-Shot Speech Generation**: We assess the content consistency (WER) and speaker similarity (SIM) of our model in zero-shot speech generation on SEED (Anastassiou et al. 2024).

- **Multilingual Speech Generation**: We assess the content consistency and speaker similarity of our model in zero-shot multilingual speech generation on MiniMax multilingual test set (Zhang et al. 2025).

- **Cross-Lingual Speech Generation**: We assess the content consistency of our model in zero-shot cross-lingual speech generation on CV3-Eval (Du et al. 2025).

### Evaluation of Zero-Shot Speech Generation

We compare the Qwen3-Omni with state-of-the-art zero-shot TTS systems. As shown in Table 3, Qwen3-Omni demonstrates highly competitive performance, highlighting its robust speech understanding and generation capabilities developed through pretraining and continual pretraining. Additionally, with reinforcement learning (RL) optimization, Qwen3-Omni yields significant improvements in generation stability, which achieves the best performance in the test-en set.

<table id="tab:zero_shot_speech_generation_table">
<caption><strong>Zero-Shot Speech Generation on Seed-TTS Test Set. The highest scores are shown in bold.</strong></caption>
<thead>
<tr>
<th style="text-align: center;"><strong>Datasets</strong></th>
<th style="text-align: left;"><strong>Model</strong></th>
<th style="text-align: left;"><strong>Performance</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="3" style="text-align: center;"><em>Content Consistency</em></td>
</tr>
<tr>
<td rowspan="9" style="text-align: center;"><table id="tab:zero_shot_speech_generation_table">
<caption><strong>Zero-Shot Speech Generation on Seed-TTS Test Set. The highest scores are shown in bold.</strong></caption>
<tbody>
<tr>
<td style="text-align: center;"><strong>SEED</strong></td>
</tr>
<tr>
<td style="text-align: center;"><em>test-zh</em> | <em>test-en</em></td>
</tr>
</tbody>
</table></td>
<td style="text-align: left;">Seed-TTS<sub>ICL</sub> (Anastassiou et al. 2024)</td>
<td style="text-align: left;">1.11 | 2.24</td>
</tr>
<tr>
<td style="text-align: left;">Seed-TTS<sub>RL</sub> (Anastassiou et al. 2024)</td>
<td style="text-align: left;">1.00 | 1.94</td>
</tr>
<tr>
<td style="text-align: left;">MaskGCT (Y. Wang et al. 2024)</td>
<td style="text-align: left;">2.27 | 2.62</td>
</tr>
<tr>
<td style="text-align: left;">E2 TTS (Eskimez et al. 2024)</td>
<td style="text-align: left;">1.97 | 2.19</td>
</tr>
<tr>
<td style="text-align: left;">F5-TTS (Yushen Chen et al. 2024)</td>
<td style="text-align: left;">1.56 | 1.83</td>
</tr>
<tr>
<td style="text-align: left;">Spark TTS (X. Wang et al. 2025)</td>
<td style="text-align: left;">1.20 | 1.98</td>
</tr>
<tr>
<td style="text-align: left;">CosyVoice 2 (Du et al. 2024)</td>
<td style="text-align: left;">1.45 | 2.57</td>
</tr>
<tr>
<td style="text-align: left;">CosyVoice 3 (Du et al. 2025)</td>
<td style="text-align: left;"><strong>0.71</strong> | 1.45</td>
</tr>
<tr>
<td style="text-align: left;">Qwen2.5-Omni-7B (Xu et al. 2025)</td>
<td style="text-align: left;">1.42 | 2.33</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: left;">Qwen3-Omni-30B-A3B</td>
<td style="text-align: left;">1.07 | <strong>1.39</strong></td>
</tr>
</tbody>
</table>

### Evaluation of Multilingual Speech Generation

Qwen3-Omni supports speech generation across 10 languages. We evaluate its performance against both the MiniMax-Speech and ElevenLabs Multilingual v2 models for multilingual speech generation. As shown in Table 6, Qwen3-Omni surpasses these models by a significant margin for languages such as Chinese, English, and French, while delivering competitive results in the remaining languages. These findings indicate that Qwen3-Omni generates cloned speech with consistent stability and human-like voice across all evaluated languages.

<table id="tab:multilingual_speech_generation_table">
<caption><strong>Multilingual Speech Generation on MiniMax Multilingual Test Set. The highest scores are shown in bold.</strong></caption>
<thead>
<tr>
<th style="text-align: left;"><strong>Language</strong></th>
<th colspan="3" style="text-align: center;"><strong>Content Consistency</strong></th>
<th colspan="3" style="text-align: center;"><strong>Speaker Similarity</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">2-7</td>
<td style="text-align: left;"><table id="tab:multilingual_speech_generation_table">
<caption><strong>Multilingual Speech Generation on MiniMax Multilingual Test Set. The highest scores are shown in bold.</strong></caption>
<tbody>
<tr>
<td style="text-align: center;"><strong>Qwen3-Omni</strong></td>
</tr>
<tr>
<td style="text-align: center;"><strong>-30B-A3B</strong></td>
</tr>
</tbody>
</table></td>
<td style="text-align: left;"><strong>MiniMax</strong></td>
<td style="text-align: left;"><strong>ElevenLabs</strong></td>
<td style="text-align: left;"><table id="tab:multilingual_speech_generation_table">
<caption><strong>Multilingual Speech Generation on MiniMax Multilingual Test Set. The highest scores are shown in bold.</strong></caption>
<tbody>
<tr>
<td style="text-align: center;"><strong>Qwen3-Omni</strong></td>
</tr>
<tr>
<td style="text-align: center;"><strong>-30B-A3B</strong></td>
</tr>
</tbody>
</table></td>
<td style="text-align: left;"><strong>MiniMax</strong></td>
<td style="text-align: left;"><strong>ElevenLabs</strong></td>
</tr>
<tr>
<td style="text-align: left;">Chinese</td>
<td style="text-align: left;"><strong>0.716</strong></td>
<td style="text-align: left;">2.252</td>
<td style="text-align: left;">16.026</td>
<td style="text-align: left;">0.772</td>
<td style="text-align: left;"><strong>0.780</strong></td>
<td style="text-align: left;">0.677</td>
</tr>
<tr>
<td style="text-align: left;">English</td>
<td style="text-align: left;"><strong>1.069</strong></td>
<td style="text-align: left;">2.164</td>
<td style="text-align: left;">2.339</td>
<td style="text-align: left;"><strong>0.773</strong></td>
<td style="text-align: left;">0.756</td>
<td style="text-align: left;">0.613</td>
</tr>
<tr>
<td style="text-align: left;">German</td>
<td style="text-align: left;">0.777</td>
<td style="text-align: left;">1.906</td>
<td style="text-align: left;"><strong>0.572</strong></td>
<td style="text-align: left;"><strong>0.738</strong></td>
<td style="text-align: left;">0.733</td>
<td style="text-align: left;">0.614</td>
</tr>
<tr>
<td style="text-align: left;">Italian</td>
<td style="text-align: left;"><strong>1.067</strong></td>
<td style="text-align: left;">1.543</td>
<td style="text-align: left;">1.743</td>
<td style="text-align: left;"><strong>0.742</strong></td>
<td style="text-align: left;">0.699</td>
<td style="text-align: left;">0.579</td>
</tr>
<tr>
<td style="text-align: left;">Portuguese</td>
<td style="text-align: left;">1.872</td>
<td style="text-align: left;">1.877</td>
<td style="text-align: left;"><strong>1.331</strong></td>
<td style="text-align: left;">0.770</td>
<td style="text-align: left;"><strong>0.805</strong></td>
<td style="text-align: left;">0.711</td>
</tr>
<tr>
<td style="text-align: left;">Spanish</td>
<td style="text-align: left;">1.765</td>
<td style="text-align: left;"><strong>1.029</strong></td>
<td style="text-align: left;">1.084</td>
<td style="text-align: left;">0.744</td>
<td style="text-align: left;"><strong>0.762</strong></td>
<td style="text-align: left;">0.615</td>
</tr>
<tr>
<td style="text-align: left;">Japanese</td>
<td style="text-align: left;">3.631</td>
<td style="text-align: left;"><strong>3.519</strong></td>
<td style="text-align: left;">10.646</td>
<td style="text-align: left;">0.763</td>
<td style="text-align: left;"><strong>0.776</strong></td>
<td style="text-align: left;">0.738</td>
</tr>
<tr>
<td style="text-align: left;">Korean</td>
<td style="text-align: left;"><strong>1.670</strong></td>
<td style="text-align: left;">1.747</td>
<td style="text-align: left;">1.865</td>
<td style="text-align: left;"><strong>0.778</strong></td>
<td style="text-align: left;">0.776</td>
<td style="text-align: left;">0.700</td>
</tr>
<tr>
<td style="text-align: left;">French</td>
<td style="text-align: left;"><strong>2.505</strong></td>
<td style="text-align: left;">4.099</td>
<td style="text-align: left;">5.216</td>
<td style="text-align: left;"><strong>0.689</strong></td>
<td style="text-align: left;">0.628</td>
<td style="text-align: left;">0.535</td>
</tr>
<tr>
<td style="text-align: left;">Russian</td>
<td style="text-align: left;">3.986</td>
<td style="text-align: left;">4.281</td>
<td style="text-align: left;"><strong>3.878</strong></td>
<td style="text-align: left;">0.759</td>
<td style="text-align: left;"><strong>0.761</strong></td>
<td style="text-align: left;">0.676</td>
</tr>
</tbody>
</table>

### Evaluation of Cross-Lingual Speech Generation

Qwen3-Omni supports not only multilingual voice cloning but also cross-lingual voice cloning. We evaluate its performance against CosyVoice2 and CosyVoice3 for cross-lingual speech generation. As shown in Table 7, Qwen3-Omni outperforms CosyVoice3 in any-to-en (any language to English) and any-to-ko (any language to Korean) voice cloning. Notably, in any-to-ja (any language to Japanese) tasks, Qwen3-Omni achieves comparable performance to CosyVoice3 even without text normalization, despite CosyVoice3 converting all Japanese characters into phonetic kana. These results highlight Qwen3-Omni’s superiority in cross-lingual speech generation, demonstrating its adaptability across diverse linguistic contexts.

| **Language** | **Qwen3-Omni-30B-A3B** | **CosyVoice3** | **CosyVoice2** |
|:-------------|:----------------------:|:--------------:|:--------------:|
| en-to-zh     |          5.37          |    **5.09**    |      13.5      |
| ja-to-zh     |          3.32          |    **3.05**    |      48.1      |
| ko-to-zh     |        **0.99**        |      1.06      |      7.70      |
| zh-to-en     |        **2.76**        |      2.98      |      6.47      |
| ja-to-en     |        **3.31**        |      4.20      |      17.1      |
| ko-to-en     |        **3.34**        |      4.19      |      11.2      |
| zh-to-ja     |          8.29          |    **7.08**    |      13.1      |
| en-to-ja     |          7.53          |    **6.80**    |      14.9      |
| ko-to-ja     |          4.24          |    **3.93**    |      5.86      |
| zh-to-ko     |        **5.13**        |      14.4      |      24.8      |
| en-to-ko     |        **4.96**        |      5.87      |      21.9      |
| ja-to-ko     |        **6.23**        |      7.92      |      21.5      |

**Cross-Lingual Speech Generation on CosyVoice3 Cross-Lingual Test Set. The highest scores are shown in bold.** {#tab:cross_lingual_speech_generation_table}

# Evaluating Non‑Degradation Across Modalities

A standardized data integration methodology is rendered impractical by the heterogeneous nature of different modalities, each requiring distinct pre-training objectives and optimization techniques. To ensure a fair and rigorous evaluation, we therefore designed a controlled comparative study. Our approach involved pre-training three models with matched parameter counts: a text-only baseline, a vision-only baseline, and a multimodal “Omni” model. To isolate the effects of multimodality, all confounding variables were meticulously controlled. Specifically, the Omni model was trained on the identical text and vision corpora as the unimodal baselines. Moreover, we aligned critical training parameters across all models, including learning rate schedules, batch sizes, and the effective number of training epochs for each modality, which was normalized by adjusting data sampling ratios. Consequently, the sole differentiating factor in our experiment was the Omni model’s inclusion of supplementary audio and audio-visual data during its pre-training phase.

The results are shown in Table Table 16, we evaluate comprehensive benchmarks covering a variety of modalities, including the text modality (general tasks, math & STEM tasks, coding tasks, multilingual tasks), the visual modality (college-level problems, OCR-related tasks), and the video modality (video understanding tasks). The experimental results not only demonstrate that mixing unimodal and cross-modal data during the early stage of text pretraining can achieve better performance across all modalities, but also indicate that joint multimodal training enables mutual enhancement between different modalities, leading to improved performance in single modalities as well. This fully showcases the versatility and robustness of Qwen3-Omni across diverse evaluation criteria.

Due to the prohibitive experimental cost, we could not conduct a comprehensive sweep across all model scales. Based on Table Table 16 and our internal experiments, we observe: (1) early multimodal integration during pretraining allows language models to be co-trained with vision or audio without any degradation in language capability; (2) the inclusion of the text modality substantially improves performance in the vision and audio. In constrast, we do not observe measurable gains in language ability from adding visual or audio signals; (3) empirically, adding audio data consistently improves vision performance on the MMMU benchmark and OCR-related tasks

# Conclusion

In this paper, we introduce Qwen3-Omni-30B-A3B, Qwen3-Omni-30B-A3B-Thinking, Qwen3-Omni-Flash-Instruct, and Qwen3-Omni-Flash-Thinking models. Qwen3-Omni-30B-A3B matches or surpasses the latest same-size unimodal Qwen models on text and vision benchmarks. Notably, on audio processing and dialogue benchmarks, it attains state-of-the-art performance among open-source systems on 32 benchmarks and is comparable to, or better than, the strong proprietary counterpart Gemini-2.5-Pro. The Qwen3-Omni-30B-A3B Thinking variant achieves further gains on complex tasks spanning text, vision, and audio-visual reasoning. Beyond accuracy, the model supports 119 text languages, 19 languages for speech recognition and 10 languages for speech synthesis, and enables audio understanding and interactive sessions up to 40 minutes. Thanks to its streaming architecture and multi-codebook design, Qwen3-Omni at the 30B-A3B scale still delivers an end-to-end first-packet latency of 234 ms.

Research fields often cycle between specialization and integration. In this context, we believe Qwen3-Omni represents a milestone: to our knowledge, it provides the first evidence that fully integrated, end-to-end multimodal training can be achieved without degrading core language capability and other modalities. We are eager to share these findings with the community and hope they will stimulate further research.

For practical usage, Qwen3-Omni-30B-A3B offers strong text and vision capabilities, robust and reliable ASR, interactive speech support in over 20 languages, very low first-packet latency for interactive use, and stable, naturalistic speech synthesis. Crucially, it exhibits advantages over cascaded pipelines, including stronger cross-modal reasoning, lower end-to-end latency, and lower system complexity and cost. In future work, we will further advance the model along multiple axes, including multi-speaker ASR, video OCR, audiovisual proactive learning, and enhanced support for agent-based workflows and function calling.

# Authors

**Core Contributors:** Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, Junyang Lin

**Contributors[^1]:** An Yang, Anfeng Li, Bei Chen, Beichen Zhang, Bin Lin, Binyuan Hui, Bohan Wang, Buxiao Wu, Chenfei Wu, Cheng Chen, Chen Qiang, Chenhan Yuan, Chenhao Li, Chenxu Lv, Chujie Zheng, Daren Chen, Dayiheng Liu, Dake Guo, Fei Huang, Gezhengyang Zhu, Guangdong Zhou, Hang Zhang, Hongjian Tu, Humen Zhong, Jialong Zuo, Jianhong Tu, Jianwei Zhang, Jiayi Leng, Jing Zhou, Jingren Zhou, Kai Dang, Kexin Yang, Kun Yan, Laiwen Zheng, Lei Xie, Lianghao Deng, Lingchen Meng, Mei Li, Miao Hong, Mingfeng Xue, Minsheng Li, Mingze Li, Peiyang Zhang, Peng Liu, Pengfei Wang, Ruibin Yuan, Rui Hu, Ruiyang Xu, Qidong Huang, Qin Zhu, Que Shen, Shen Li, Shixuan Liu, Sibo Song, Siqi Zhang, Song Chen, Su Hao, Tianyi Tang, Wenbin Ge, Wentao Yao, Wei Ding, Wei Wang, Xiaodong Deng, Xiaotong Chen, Xiao Li, Xian Yang, Xinyao Niu, Xudong Guo, Xin Le, Xuechun Wang, Xutong Jin, Xuancheng Ren, Yang Fan, Yang Liu, Yang Su, Yantao Liu, Yi Wu, Yichang Zhang, Yilei Chen, Yiming Dong, Yinger Zhang, Yizhong Cao, Yuchong Sun, Yuezhang Wang, Yuhao Wang, Yuqiong Liu, Yuanzhi Zhu, Yuxiang Chen, Yuxuan Cai, Yuxuan Liu, Zeyu Cui, Zheng Li, Zhenghao Xing, Zhenru Zhang, Zihan Qiu, ZiYue Jiang, Zhaohai Li, Zhi Li, Zhibo Yang, Zhihai Wang, Zhipeng Zhou

# Appendix

## More Evaluation on Speech and Music Understanding

This section reports the performance of the Qwen3-Omni-thinking model on tasks pertaining to ASR/S2TT and Music. As shown in Table Table 17 and Table 18, in the domains of ASR/S2TT and Music understanding, the Qwen3-Omni-Thinking model is outperformed by its Instruct counterpart, which indicates that for these predominantly perception-based tasks, the engagement of sophisticated reasoning processes fails to yield performance gains. In fact, it may even introduce a higher propensity for hallucinations.

## Qualitative Results from Qwen3-Omni-30B-A3B-Captioner

In this section, we illustrate the performance of our finetuned Qwen3-Omni-30B-A3B-Captioner through three representative case studies. The selected scenarios are designed to test the model’s proficiency in: (1) analyzing expressive speech, (2) interpreting complex auditory scenes and sound effects, and (3) captioning composite audio that includes speech, music, and ambient sounds. For reproducibility, the original audio samples can be accessed from our public GitHub repository.

UTF8gkai

The audio clip opens in a studio setting, marked by a faint, persistent electronic hiss and a subtle low-frequency hum, indicating professional but not entirely noise-free recording conditions. The male speaker, whose voice is delivered in a clear, energetic, and highly theatrical manner, begins with an assertive “对” (“Right!”), delivered with a sharp, rising intonation that immediately establishes a performative, self-aware tone. He continues, “这就是我，万人敬仰的太乙真人” (“This is me, Taiyi Zhenren, admired by thousands”), using exaggerated emphasis and a pronounced rising inflection on “万人敬仰” (“admired by thousands”) to convey mock grandeur and comic bravado. His articulation is crisp, and his voice is rich in midrange frequencies, suggesting close-mic’d recording with a touch of digital reverb for dramatic effect.\
After a brief, deliberate pause, the speaker shifts to a more self-deprecating yet still playful register: “虽然有点婴儿肥，但也掩不住我逼人的帅气” (“Although I’m a bit chubby, it still can’t hide my impressive handsomeness”). Here, the tone becomes lighter and more conversational, with a subtle downward inflection on “婴儿肥” (“chubby”) and a playful, rising emphasis on “逼人的帅气” (“impressive handsomeness”), underscoring the comedic contrast between modesty and self-praise. The delivery is marked by expressive inflection and clear articulation, maintaining a consistent presence in the mix.\
Throughout the speech, the recording remains clean and undistorted, with no background noises, environmental cues, or audience sounds. The voice is centrally positioned in the stereo field, and the use of reverb and EQ creates a sense of theatricality and spatial depth, while the electronic hiss and hum are subtle but present, hinting at professional studio equipment and post-processing. The clip concludes with a sudden, complete cut to silence, with no trailing sounds or fade-out, indicating a deliberate and precise edit.\
In summary, this audio features a single male speaker delivering a self-introductory monologue in Mandarin, portraying the legendary Taoist figure Taiyi Zhenren in a comedic, over-the-top manner. The performance blends exaggerated bravado with self-deprecating humor, employing dramatic inflection, expressive pacing, and professional studio production to create a memorable, entertaining character portrayal. The absence of background noise and the clean, intentional editing reinforce the clip’s crafted, theatrical nature, making it suitable for online content, podcasts, or digital entertainment aimed at a modern, culturally literate audience.

UTF8gkai

The audio clip is a highly produced, cinematic soundscape, lasting 25 seconds, designed to evoke intense drama and imminent danger. It begins with a deep, resonant musical drone and a sharp, metallic clank, immediately establishing a mood of tension and foreboding. A slow, rhythmic, and ominous beat joins, layered with swelling orchestral strings and synthetic textures that build suspense and a sense of scale.

Around the 9-second mark, the soundscape explodes with the thunderous, mechanical roar of a massive engine, its deep, percussive pulses dominating the mix. This is accompanied by a high-pitched, metallic screech, suggesting immense machinery in motion or under strain. The orchestral music intensifies, becoming more urgent and climactic, with the addition of a high-frequency electronic whine that adds to the feeling of technological threat.\
At the 15-second point, a colossal, explosive impact occurs, characterized by a sharp, high-frequency crack followed by a deep, resonant boom. The sound of shattering and debris fills the space, with reverberation suggesting a vast, hard-walled environment—perhaps a hangar, tunnel, or industrial facility. The orchestral music reaches its peak, becoming chaotic and overwhelming, before abruptly ceasing at the 19-second mark.\
Immediately after the impact, the focus shifts to the sounds of a single person: heavy, strained breathing, the rustle of heavy fabric, and a low groan. These sounds, recorded with close-mic clarity and natural reverberation, indicate the individual is recovering from exertion or injury, likely amidst the aftermath of the catastrophic event. There are no spoken words, but the vocalizations and physical sounds strongly suggest the presence of a male character.\
Throughout the clip, there are no voices, dialogue, or environmental cues that would suggest a specific culture, language, or geographic setting. The orchestral and electronic musical elements are universal to modern action, sci-fi, and thriller genres, and the production values are consistent with contemporary Hollywood or AAA video game trailers.\
In summary, the audio is a tightly crafted, non-verbal scene of a catastrophic event—likely a crash, explosion, or mechanical disaster—in a vast, industrial or futuristic setting, immediately followed by the physical and emotional aftermath of a single survivor. The absence of speech, combined with the orchestration, sound design, and spatial cues, positions the clip as a generic but highly effective trailer or scene cue for a dramatic, action-oriented media product.\

UTF8gkai

The audio begins with a deep, resonant metallic clang, immediately followed by a powerful, low-frequency rumble that evokes the sound of a massive engine or heavy machinery. The space is acoustically vast, with long reverberation tails and a faint, high-pitched whine, suggesting a colossal, metallic environment such as a spaceship hangar or a vast industrial chamber. A mechanical whirring sound emerges, layered with metallic clanking and grinding, creating the impression of a large vehicle or portal activating. These sounds are punctuated by a sharp, high-frequency crackle, reminiscent of electrical arcs or energy discharges, further reinforcing the technological setting. The mechanical noises intensify, with rapid, high-pitched metallic screeches and impacts that imply the movement or collision of massive metal components.\
Suddenly, the mechanical sounds fade, and the ambient rumble persists, revealing a subtle, high-frequency hiss—possibly from an air filtration system or atmospheric turbulence—while the vastness of the space remains evident through lingering echoes. A female voice, distant and high-pitched, emerges with a plaintive, childlike tone, asking, "Are we there yet?" Her speech is slightly muffled and reverberant, indicating she is physically separated from the microphone, likely inside the vehicle or machinery. This is followed by a deeper, gravelly male voice, close to the microphone, responding with a gruff, impatient tone: "We get there when we get there." His voice is clear and assertive, contrasting with the female’s, and the exchange is typical of familial banter.\
The mechanical rumble swells again, joined by a whooshing sound as if air is rushing past, and a rapid metallic clatter signals the rapid movement of machinery or vehicles. The environment is further emphasized by a sharp, high-frequency crackle, suggesting an energy surge or system overload. A third male voice, energetic and friendly, calls out from a moderate distance: "How you doing, honey?" His tone is warm and affectionate, with a slight echo, and the use of "honey" implies a familial relationship. Immediately after, the female voice, now closer and more urgent, responds with a high-pitched, exasperated tone: "Do I have to answer?" Her delivery is quick, sharp, and filled with playful annoyance, reflecting a familiar and comfortable dynamic among the group.\
As the mechanical sounds subside, a low-frequency hum remains, and the audio transitions into a brief, synthesized musical sting. This consists of a single sustained note from a low-frequency synthesizer, likely a bass or synth pad, which is cut off abruptly, suggesting the end of the scene or a transition to another segment. Throughout, the audio is of high fidelity, with no distortion or noise, and each sound is distinct and well-defined. The spatial characteristics—distance, direction, and reverberation—contribute to a vivid sense of a large, metallic, and technological environment. The dialogue is clear and expressive, with emotional tones ranging from impatience and warmth to playful annoyance. The use of "honey" and the familial banter reinforce the impression of a close-knit group, likely family members, engaged in a shared journey within a science fiction or fantasy context.\
In summary, the audio presents a dynamic, high-fidelity soundscape of a massive, metallic environment—possibly a spaceship or futuristic vehicle—where a group of family members engage in playful banter as they travel together. Mechanical sounds, spatial cues, and expressive dialogue combine to create a vivid sense of place and character, culminating in a synthesized musical sting that signals a narrative transition. The scene is rich in emotional nuance and technological detail, firmly situating the listener within a science fiction or fantasy setting.

Agrawal, Pravesh, Szymon Antoniak, Emma Bou Hanna, et al. 2024. *Pixtral 12B*. <https://arxiv.org/abs/2410.07073>.

AIME. 2025. *AIME Problems and Solutions*. <https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions>.

Anastassiou, Philip, Jiawei Chen, Jitong Chen, et al. 2024. “Seed-TTS: A Family of High-Quality Versatile Speech Generation Models.” *arXiv Preprint arXiv:2406.02430*.

Anthropic. 2023a. *Claude 2*. Anthropic. <https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf>.

Anthropic. 2023b. *Introducing Claude*. Anthropic. <https://www.anthropic.com/index/introducing-claude>.

Anthropic. 2024. *The Claude 3 Model Family: Opus, Sonnet, Haiku*. Anthropic, AI. [https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model\\Card\\Claude\\3.pdf](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model/_Card/_Claude/_3.pdf).

Bai, Jinze, Shuai Bai, Yunfei Chu, et al. 2023. “Qwen Technical Report.” *CoRR* abs/2309.16609.

Bai, Jinze, Shuai Bai, Shusheng Yang, et al. 2023. “Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities.” *CoRR* abs/2308.12966.

Bai, Shuai, Keqin Chen, Xuejing Liu, et al. 2025. “Qwen2. 5-Vl Technical Report.” *arXiv Preprint arXiv:2502.13923*.

Bogdanov, Dmitry, Minz Won, Philip Tovstogan, Alastair Porter, and Xavier Serra. 2019. “The Mtg-Jamendo Dataset for Automatic Music Tagging.”

Brown, Tom, Benjamin Mann, Nick Ryder, et al. 2020. “Language Models Are Few-Shot Learners.” *NeurIPS*.

Cassano, Federico, John Gouwar, Daniel Nguyen, et al. 2023. “MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation.” *IEEE Trans. Software Eng.* 49 (7): 3675–91.

Chen, Lin, Jinsong Li, Xiaoyi Dong, et al. 2024. “Are We on the Right Way for Evaluating Large Vision-Language Models?” *arXiv:2403.20330*.

Chen, Yiming, Xianghu Yue, Chen Zhang, Xiaoxue Gao, Robby T Tan, and Haizhou Li. 2024. “Voicebench: Benchmarking Llm-Based Voice Assistants.” *arXiv Preprint arXiv:2410.17196*.

Chen, Yushen, Zhikang Niu, Ziyang Ma, et al. 2024. “F5-Tts: A Fairytaler That Fakes Fluent and Faithful Speech with Flow Matching.” *arXiv Preprint arXiv:2410.06885*.

Cheng, Junhao, Yuying Ge, Teng Wang, Yixiao Ge, Jing Liao, and Ying Shan. 2025. “Video-Holmes: Can MLLM Think Like Holmes for Complex Video Reasoning?” *CoRR* abs/2505.21374.

Chu, Yunfei, Jin Xu, Qian Yang, et al. 2024. “Qwen2-Audio Technical Report.” *arXiv Preprint arXiv:2407.10759*.

Chu, Yunfei, Jin Xu, Xiaohuan Zhou, et al. 2023. “Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models.” *CoRR* abs/2311.07919.

Comanici, Gheorghe, Eric Bieber, Mike Schaekermann, et al. 2025. “Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities.” *arXiv Preprint arXiv:2507.06261*.

Du, Zhihao, Changfeng Gao, Yuxuan Wang, et al. 2025. “CosyVoice 3: Towards in-the-Wild Speech Generation via Scaling-up and Post-Training.” *CoRR* abs/2505.17589.

Du, Zhihao, Yuxuan Wang, Qian Chen, et al. 2024. “CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models.” *arXiv Preprint arXiv:2412.10117*.

Dubey, Abhimanyu, Abhinav Jauhri, Abhinav Pandey, et al. 2024. “The Llama 3 Herd of Models.” *CoRR* abs/2407.21783.

Eskimez, Sefik Emre, Xiaofei Wang, Manthan Thakker, et al. 2024. “E2 Tts: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot Tts.” *2024 IEEE Spoken Language Technology Workshop (SLT)*, 682–89.

Fu, Chaoyou, Yuhan Dai, Yondong Luo, et al. 2024. “Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-Modal LLMs in Video Analysis.” *arXiv:2405.21075*.

Gema, Aryo Pradipta, Joshua Ong Jun Leang, Giwon Hong, et al. 2024. “Are We Done with MMLU?” *CoRR* abs/2406.04127.

Gemini Team. 2024. *Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context*. Google. [https://storage.googleapis.com/deepmind-media/gemini/gemini\\v1\\5\\report.pdf](https://storage.googleapis.com/deepmind-media/gemini/gemini/_v1/_5/_report.pdf).

Guan, Tianrui, Fuxiao Liu, Xiyang Wu, et al. 2024. “Hallusionbench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models.” *IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024*, 14375–85.

He, Yun, Di Jin, Chaoqi Wang, et al. 2024. “Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following.” *CoRR* abs/2410.15553. <https://doi.org/10.48550/ARXIV.2410.15553>.

Hong, Jack, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. 2025. “WorldSense: Evaluating Real-World Omnimodal Understanding for Multimodal LLMs.” *CoRR* abs/2502.04326.

Kembhavi, Aniruddha, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. 2016. “A Diagram Is Worth a Dozen Images.” *ECCV*.

Kwon, Woosuk, Zhuohan Li, Siyuan Zhuang, et al. 2023. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*.

Law, Edith, Kris West, Michael I Mandel, Mert Bay, and J Stephen Downie. 2009. “Evaluation of Algorithms Using Games: The Case of Music Tagging.” *ISMIR*, 387–92.

Li, Junnan, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. “Blip-2: Bootstrapping Language-Image Pre-Training with Frozen Image Encoders and Large Language Models.” *arXiv:2301.12597*.

Lin, Bill Yuchen, Ronan Le Bras, Kyle Richardson, et al. 2025. “ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning.” *CoRR* abs/2502.01100.

Liu, Haotian, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. “Visual Instruction Tuning.” *arXiv:2304.08485*.

Lu, Pan, Hritik Bansal, Tony Xia, et al. 2024. “MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts.” *ICLR*.

Masry, Ahmed, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. “ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning.” *arXiv:2203.10244*.

OpenAI. 2022. *ChatML*. <https://github.com/openai/openai-python/blob/e389823ba013a24b4c32ce38fa0bd87e6bccae94/chatml.md>.

OpenAI. 2023. “GPT4 Technical Report.” *CoRR* abs/2303.08774.

OpenAI. 2024. *Hello GPT-4o*. <https://openai.com/index/hello-gpt-4o/>.

Paech, Samuel J. 2024. *Creative Writing V3*. <https://eqbench.com/creative_writing.html>.

Paiss, Roni, Ariel Ephrat, Omer Tov, et al. 2023. “Teaching CLIP to Count to Ten.” *IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023*, 3147–57.

Rafailov, Rafael, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. “Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.” *NeurIPS*.

Rein, David, Betty Li Hou, Asa Cooper Stickland, et al. 2023. “GPQA: A Graduate-Level Google-Proof Q&A Benchmark.” *CoRR* abs/2311.12022.

Sakshi, S, Utkarsh Tyagi, Sonal Kumar, et al. 2024. *MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark*. <https://arxiv.org/abs/2410.19168>.

Su, Jianlin, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. “RoFormer: Enhanced Transformer with Rotary Position Embedding.” *Neurocomputing* 568: 127063.

Touvron, Hugo, Louis Martin, Kevin Stone, et al. 2023. “Llama 2: Open Foundation and Fine-Tuned Chat Models.” *arXiv:2307.09288*.

Tschannen, Michael, Alexey Gritsenko, Xiao Wang, et al. 2025. “SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.” *Https://Arxiv.org/Abs/2502.14786*.

Tzanetakis, George, and Perry Cook. 2002. “Musical Genre Classification of Audio Signals.” *IEEE Transactions on Speech and Audio Processing* 10 (5): 293–302.

Wang, Dingdong, Jincenzi Wu, Junan Li, et al. 2025. “MMSU: A Massive Multi-Task Spoken Language Understanding and Reasoning Benchmark.” *CoRR* abs/2506.04779. <https://doi.org/10.48550/ARXIV.2506.04779>.

Wang, Ke, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. 2024. “Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset.” *arXiv:2402.14804*.

Wang, Weihan, Zehai He, Wenyi Hong, et al. 2024. “LVBench: An Extreme Long Video Understanding Benchmark.” *CoRR* abs/2406.08035.

Wang, Xinsheng, Mingqi Jiang, Ziyang Ma, et al. 2025. “Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens.” *CoRR* abs/2503.01710.

Wang, Yiming, Pei Zhang, Jialong Tang, et al. 2025. “PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts.” *CoRR* abs/2504.18428. <https://doi.org/10.48550/ARXIV.2504.18428>.

Wang, Yuancheng, Haoyue Zhan, Liwei Liu, et al. 2024. “Maskgct: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer.” *arXiv Preprint arXiv:2409.00750*.

Wu, Yuning, Jiahao Mei, Ming Yan, et al. 2025. “WritingBench: A Comprehensive Benchmark for Generative Writing.” *CoRR* abs/2503.05244.

Xu, Jin, Zhifang Guo, Jinzheng He, et al. 2025. “Qwen2. 5-Omni Technical Report.” *arXiv Preprint arXiv:2503.20215*.

Yan, Fanjia, Huanzhi Mao, Charlie Cheng-Jie Ji, et al. 2024. *Berkeley Function Calling Leaderboard*. <a href="https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html" class="uri">Https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html</a>.

Yang, An, Anfeng Li, Baosong Yang, et al. 2025. “Qwen3 Technical Report.” *arXiv Preprint arXiv:2505.09388*.

Yang, An, Baosong Yang, Binyuan Hui, et al. 2024. “Qwen2 Technical Report.” *arXiv:2407.10671*.

Yuan, Ruibin, Yinghao Ma, Yizhi Li, et al. 2023. “Marble: Music Audio Representation Benchmark for Universal Evaluation.” *Advances in Neural Information Processing Systems* 36: 39626–47.

Yue, Xiang, Yuansheng Ni, Kai Zhang, et al. 2023. “Mmmu: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert Agi.” *arXiv:2311.16502*.

Yue, Xiang, Tianyu Zheng, Yuansheng Ni, et al. 2024. “MMMU-Pro: A More Robust Multi-Discipline Multimodal Understanding Benchmark.” *arXiv Preprint arXiv:2409.02813*.

Zang, Yongyi, Sean O’Brien, Taylor Berg-Kirkpatrick, Julian McAuley, and Zachary Novack. 2025. “Are You Really Listening? Boosting Perceptual Awareness in Music-Qa Benchmarks.” *arXiv Preprint arXiv:2504.00369*.

Zhang, Bowen, Congchao Guo, Geng Yang, et al. 2025. “MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder.” *CoRR* abs/2505.07916.

Zheng, Chujie, Shixuan Liu, Mingze Li, et al. 2025. “Group Sequence Policy Optimization.” *arXiv Preprint arXiv:2507.18071*.

Zhou, Jeffrey, Tianjian Lu, Swaroop Mishra, et al. 2023. “Instruction-Following Evaluation for Large Language Models.” *CoRR* abs/2311.07911.

Zhou, Junjie, Yan Shu, Bo Zhao, et al. 2025. “MLVU: Benchmarking Multi-Task Long Video Understanding.” *IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025*, 13691–701.

Zhou, Ziwei, Rui Wang, and Zuxuan Wu. 2025. “Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment Across Modalities.” *CoRR* abs/2505.17862.

Zhu, Deyao, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. “Minigpt-4: Enhancing Vision-Language Understanding with Advanced Large Language Models.” *arXiv:2304.10592*.

[^1]: Alphabetical order.
