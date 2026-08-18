# Homework: Evidence-Based Video Evaluation

## Goal

Evaluate two live analyses of the supplied NASA video at the syntax, schema, temporal, and visible-evidence layers. Then revise the prompt once and test whether the change addresses a specific observed failure. Expected working time: **75–90 minutes**.

## Before you begin

- Connect to the Clemson network or CUVPN.
- Complete `notebooks/00_setup_and_diagnostics.ipynb` and Lecture 03.
- Use the supplied `assets/nasa_three_pacific_hurricanes.mp4` only.
- Use `qwen3-omni-30b-a3b` for both required analyses.
- Read `assets/video_ground_truth.json` and its `annotation_policy` before scoring.
- Do not upload private, student, or unapproved video.

Create `03_video_analysis_homework_submission.ipynb`.

## Part A — Validate the media and reference

1. Run `validate_video_path()` on the NASA MP4.
2. Record the resolved filename, MIME type, byte size, and the 44-second reference duration.
3. Read the three human-authored reference events.
4. In one sentence, explain why storm names and categories are metadata rather than pixel-only observations.
5. Do not modify the reference after seeing model output.

## Part B — Run two model configurations

Use the same video, prompt, model, and `temperature=0.1` in both conditions:

1. **Non-thinking:** `enable_thinking=False`, `max_tokens=1200`.
2. **Thinking:** `enable_thinking=True`, `max_tokens=4096`.

For each condition:

1. Save the returned analysis object.
2. Confirm that the top level contains `summary`, `events`, and `uncertainties`.
3. Run `validate_video_analysis()`.
4. Record whether parsing and schema validation passed.
5. Count predicted events and list the uncertainty statements.

The larger thinking budget is required because reasoning and final JSON share the newly generated token allowance. Do not interpret a longer response as evidence of correctness.

## Part C — Audit time and visible support

For each configuration, create one row for every reference event and every unmatched predicted event:

| condition | reference event | predicted event | temporal IoU | match category | visible support | reviewer note |
|---|---|---|---:|---|---|---|

Use `temporal_iou()` to identify candidate temporal matches. Then make a separate human judgment about the description and evidence.

Use these categories:

- **matched:** reasonable temporal overlap and compatible visible event;
- **missing:** a reference event has no suitable prediction;
- **extra:** a prediction has no reference counterpart;
- **shifted:** the event is plausible but its time range is materially displaced;
- **unsupported:** the wording claims identity, category, intent, causality, speech, or another detail not established by visible evidence.

Answer these questions:

- Did valid JSON correspond to a fully supported answer?
- Which event was hardest to align, and why?
- Did either response use timestamps that appear more precise than the evidence allows?
- Did the thinking configuration reduce, preserve, or introduce unsupported claims?
- What can temporal IoU detect, and what must remain a semantic or human judgment?

## Part D — Revise one prompt and rerun

Choose one failure observed in Part C. Add a short instruction addressing that failure. For example:

> Describe only visible cloud systems and the day/night transition. Do not name storms, infer categories, causes, intent, or impacts. Treat timestamps as estimates and state what cannot be established visually.

1. Append your revision to `VIDEO_ANALYSIS_PROMPT`.
2. Rerun the **non-thinking** condition only, with the same model and token budget.
3. Repeat the relevant audit rows.
4. State whether the targeted failure improved, stayed the same, or became worse.
5. Identify one tradeoff or remaining limitation. A prompt can reduce a behavior without guaranteeing its removal.

## Hints

<details>
<summary>Hint 1 — Keep structure and truth separate</summary>

`validate_video_analysis()` checks required fields, types, chronology, and valid ranges. It cannot determine whether a hurricane name, category, or causal claim is visible in the pixels.

</details>

<details>
<summary>Hint 2 — Match before interpreting</summary>

For each reference interval, calculate temporal IoU with every predicted interval and inspect the highest-overlap candidate. Do not accept the candidate until its description also matches the visible event.

</details>

<details>
<summary>Hint 3 — Treat zero overlap carefully</summary>

A zero-IoU prediction may be missing, extra, or badly shifted. Read the descriptions and inspect the video rather than deciding from the number alone.

</details>

<details>
<summary>Hint 4 — Thinking output budget</summary>

If a thinking request contains reasoning but no final JSON answer, the generated-token budget may have been exhausted. Use `max_tokens=4096` for this classroom experiment.

</details>

## What to submit

Submit the completed notebook containing:

- the local media/reference checks;
- both required analysis objects;
- parsing and schema results;
- the completed temporal and visible-support audit table;
- answers to all five Part C questions;
- the revised prompt, rerun, and before/after judgment;
- one remaining limitation;
- no private media, API key, or hidden reasoning text.

## Rubric (10 points)

- **Media, model, and structured-output checks (2):** approved asset and required configurations are used and validated.
- **Temporal evaluation (2):** candidate matches and IoU values are calculated and categorized transparently.
- **Visible-evidence audit (3):** missing, extra, shifted, and unsupported content are distinguished using the annotation policy.
- **Prompt revision experiment (2):** one observed failure motivates a controlled rerun and evidence-based comparison.
- **Responsible reporting (1):** uncertainty and limitations are stated without exposing credentials, private media, or hidden reasoning.
