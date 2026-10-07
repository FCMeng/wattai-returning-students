# Homework: Compare and Parse Hugging Face Model Outputs

**Due:** October 18, 2026

## Goal

Compare thinking and direct generation under a controlled timing contract, run one real multimodal model, and build a parser for the actual output objects you observe. Expected working time: **90–120 minutes** plus model-download time.

## Before you begin

Run `notebooks/00_setup_and_diagnostics.ipynb` and complete Lecture 01. Use Palmetto 2 with a suitable CUDA GPU. Record the GPU model, dtype, package versions, and whether `device_map="auto"` placed any layers on CPU or disk. Do not substitute hand-written text for model output.

## Part A — Audit the selected artifacts

Use `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. For the multimodal comparison, choose **one** model that fits your allocation:

- `Qwen/Qwen3-VL-8B-Instruct` at revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; or
- `Qwen/Qwen3-Omni-30B-A3B-Instruct` at revision `26291f793822fb6be9555850f06dfe95f2d7e695` only when you have the documented large-memory environment.

For every model you run, record model ID, revision, license, model class, processor class, parameter dtype, device map, weight-download size when available, and media types accepted. Explain why a tokenizer alone is insufficient for image or video input.

## Part B — Compare Qwen3.5 thinking and direct modes

Use this fixed prompt for both modes:

> A clinic has 17 appointments and 5 rooms. Explain a fair scheduling plan, then give a two-sentence final recommendation.

Load the model once. Run one untimed warm-up, then collect at least three timed runs per mode with the same `max_new_tokens`, decoding settings, prompt, and GPU. Synchronize CUDA around the timed region. Store mode, input tokens, output tokens, wall time, tokens/second, completion status, and final answer for every run. Report median latency and explain whether latency changed because of throughput, output length, or both.

Use `enable_thinking=False` for direct mode and `enable_thinking=True` for thinking mode. Do not use `/think` or `/nothink`. If the thinking output reaches the token cap without `</think>`, increase the cap once and record the second run rather than claiming that parsing failed.

## Part C — Run one real image or video interaction

For Qwen3-VL, use `assets/figures/lecture-01-01-hf_artifacts.png` and ask the model to describe the flow and identify one limitation. Print every processor-output key with tensor shape and dtype before generation.

For Qwen3-Omni, use `assets/nasa_three_pacific_hurricanes.mp4`, keep `return_audio=False`, and ask the model to list visible weather systems and state what the pixels alone cannot prove. Record the returned tuple/object types and whether an audio value is present. Do not attempt Omni on a GPU allocation below the model card's documented requirement.

## Part D — Discover and normalize the output format

For each actual run, save:

1. `type(raw_output).__name__`;
2. a safely truncated `repr(raw_output)`;
3. mapping keys or public attributes;
4. the exact path used to reach token IDs, decoded text, reasoning text, final text, and optional audio.

Write a model/client-specific adapter that returns:

```python
{
    "final_text": str,
    "reasoning_text": str | None,
    "audio_present": bool,
    "raw_type": str,
}
```

Deliberately apply one wrong assumption—for example `.choices[0].message.content` to a local tensor, `result["generated_text"]` to an OpenAI object, or treating an Omni pair as one tensor. Capture the exception or wrong value, explain why the assumption failed, and then show the corrected parser.

## Part E — Explore one additional model

Choose another official Hugging Face chat or multimodal model that fits your hardware. Record its pinned revision and model-card URL. Determine its expected message schema and output type from official documentation, run one non-sensitive prompt, and extend your adapter or add a separate adapter. Explain why reusing one parser unchanged was or was not safe.

## Hints

<details><summary>Timing CUDA correctly</summary>Call `torch.cuda.synchronize()` immediately before starting the clock and immediately after generation. Otherwise the CPU timer may stop while GPU kernels are still queued.</details>

<details><summary>Finding the final answer</summary>Preserve the raw decoded text first. If it contains `<think>`, split at `</think>` only after checking that the closing marker exists. Missing closure usually means truncation.</details>

<details><summary>Inspecting an unfamiliar result</summary>Start with `type`, a truncated `repr`, mapping keys, and non-private attributes. Do not begin by guessing `.content`.</details>

<details><summary>Freeing GPU memory</summary>Delete the model and processor, run `import gc; gc.collect()`, then call `torch.cuda.empty_cache()`. Confirm memory before loading the next large model.</details>

## Suggested workflow

1. Request a suitable GPU and record the pinned model, processor, software, dtype, and device map.
2. Load Qwen3.5 once, warm up, and collect repeated thinking/direct timing records.
3. Run one real image or video interaction and inspect every processor tensor and raw output type.
4. Demonstrate one incorrect parsing assumption, then implement and test the corrected adapter.
5. Explore one additional official model and explain whether the existing parser remains valid.

## What to submit

Submit one notebook, `timing_records.jsonl`, your parser/adapters, the artifact table, actual image/video output, the deliberate parser failure, and a short comparison discussing latency, output length, hardware, truncation, and format differences. Do not submit model weights or sensitive chain-of-thought logs.

## Rubric (10 points)

- Pinned artifact, hardware, and processor audit (2)
- Controlled thinking/direct timing experiment (3)
- Correct real image/video interaction and tensor inspection (2)
- Raw-output investigation, failed assumption, and corrected parser (2)
- Additional-model exploration and calibrated conclusion (1)
