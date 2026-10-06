# Homework: Evaluate and Package an Adapter

## Goal

Create an evidence-backed evaluation report and a reusable adapter package. Expected working time: **75–100 minutes** after adapters are available.

## Part A — Freeze the evaluation contract

Before generating, select ten evaluation IDs by a deterministic rule and write them to `evaluation_contract.json`. Record base model ID/revision, tokenizer source, adapter path/hash, `do_sample=False`, `max_new_tokens=80`, package versions, and scoring definitions.

## Part B — Collect paired evidence

For each ID, generate base and LoRA answers from identical message histories. Store prompt ID, prompt, human reference, both outputs, latency, and output token counts in JSONL. Add blinded human labels for instruction compliance, factual concern, unsupported claim, and preferred answer. Include written criteria so another grader could repeat the labels.

## Part C — Report successes and failures

Aggregate each label with its denominator. Show at least one case where LoRA is preferred, one where base is preferred or tied, and one failure. Distinguish mechanical pipeline checks from semantic judgments. State why ten examples cannot establish broad capability improvement.

## Part D — Verify packaging

Save the adapter and tokenizer. Write a manifest containing base revision, data-manifest hash, configuration, seed, versions, metrics, and file hashes. Reload the adapter into a fresh base. Merge a copy, compare pre/post-merge logits on one fixed prompt, and report maximum absolute difference and both artifact sizes.

## Hints

<details><summary>Avoid selection bias</summary>Choose IDs before generating—such as the first ten sorted IDs or a seeded sample—and record the rule.</details>

<details><summary>Blind review</summary>Label outputs as A/B and hide the variant name until after judging.</details>

<details><summary>Merge verification</summary>Use the same tokenized prompt and evaluation mode. Tensor shapes must match; expect small floating-point differences, not exact bit identity.</details>

## Suggested workflow

1. Save evaluation IDs and decoding rules before viewing generated outputs.
2. Collect base and adapter records with the same prompt histories.
3. Blind the variant names, apply written review criteria, and aggregate denominators.
4. Show representative successes, ties, and failures without cherry-picking.
5. Write hashes and lineage, reload, merge, and report numerical verification.

## What to submit

Submit the evaluation contract, paired JSONL, scorecard, human-review criteria, adapter manifest/card, reload output, merge-equivalence result, and a deployment recommendation with limitations.

## Rubric (10 points)

- Frozen paired evaluation and traceable raw records (3)
- Transparent human criteria, aggregate results, and failures (3)
- Adapter lineage, hashes, reload, and merge verification (3)
- Calibrated deployment conclusion (1)
