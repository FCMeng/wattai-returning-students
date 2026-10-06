# Homework: Audit OpenAssistant Data and Build the SFT Objective

## Goal

Prove that the packaged OpenAssistant split is isolated and construct response-only labels with the real classroom tokenizer. Expected working time: **75–90 minutes**.

## Part A — Validate provenance and splits

Load all 2,000 training and 250 evaluation rows. Report row counts, unique `source_tree_id` counts, missing IDs, role-sequence violations, exact duplicate conversations within training, and exact duplicates across splits. Assert that train/evaluation tree overlap is zero. Show one real multi-turn example with content shortened only for display.

## Part B — Render and label one conversation

Load the pinned SmolLM2 tokenizer. Render the selected messages as text, tokenize the complete conversation, and construct labels that are `-100` for prompt tokens and active for the final assistant answer. Report `T`, the first active position, active token count, and active-label fraction. Decode only active labels and verify that they correspond to the human assistant answer.

## Part C — Audit truncation

Measure token lengths for at least 500 training rows. Report median, p90, p95, maximum, and counts above 256 and 512 tokens. Apply a 256-token limit to the longest example. State what prompt context and answer content are lost, then propose a documented filter or truncation policy.

## Part D — Deliberately break the objective

Create two incorrect label sets: one that trains on every token and one whose truncation removes every answer token. For each, explain the misleading behavior it would teach or the reason the example contributes no useful loss.

## Hints

<details><summary>Stable duplicate signature</summary>Serialize `messages` with sorted JSON keys and hash the UTF-8 bytes. Do not compare only row IDs.</details>

<details><summary>Tree isolation</summary>Build sets from `source_tree_id`; row-level random splitting is not enough because branches can share ancestors.</details>

<details><summary>Active labels</summary>`sum(label != -100 for label in labels)` counts supervised positions. Decode only those non-ignored IDs.</details>

## Suggested workflow

1. Load both packaged splits and run schema, duplicate, and tree-overlap checks.
2. Select one real conversation and render it with the pinned tokenizer.
3. Build and inspect response-only labels, including decoded active tokens.
4. Measure the real length distribution and inspect the longest example.
5. Run both broken-label tests and justify a truncation/filter policy.

## What to submit

Submit one notebook with tables, assertions, one rendered example, label inspection, length distribution, two failure demonstrations, and your recommended policy.

## Rubric (10 points)

- Provenance, schema, duplicate, and tree-isolation audit (3)
- Correct tokenizer rendering and response-only labels (3)
- Quantitative truncation analysis and justified policy (2)
- Broken examples, interpretation, and clear reporting (2)
