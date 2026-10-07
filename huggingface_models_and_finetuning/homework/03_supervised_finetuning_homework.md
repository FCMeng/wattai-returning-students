# Homework: Train and Reload a Real LoRA Adapter

**Due:** November 1, 2026

## Goal

Run a controlled PEFT experiment on real OpenAssistant records, save the adapter, and prove that a fresh process can reload it. Expected working time: **90–120 minutes** plus model download time.

## Part A — Establish the configuration

Use the pinned SmolLM2-135M-Instruct model and the first 64 training rows after validation. Reserve the final 16 of those rows for validation. Configure LoRA on `q_proj`, `k_proj`, `v_proj`, and `o_proj` with rank 8, alpha 16, dropout 0.05, and bias `none`. Record seed, maximum length, microbatch, accumulation steps, learning rate, and optimizer-step count.

## Part B — Inspect before training

Print the names of the first ten trainable parameters. Report trainable count, total count, percentage, and effective batch size. Assert that at least one parameter is trainable and that fewer than 5% of all parameters are trainable.

## Part C — Train one controlled comparison

Train a baseline adapter for at least 5 optimizer steps. Train a second adapter that changes **only rank** from 8 to 16. Store both loss histories, wall time, peak CUDA memory when available, and directory size. Do not claim the lower training loss is the better general model.

## Part D — Reload and test

In a fresh model object, reload each adapter with `is_trainable=False`. Use the same three held-out prompts and greedy `max_new_tokens=80`. Save outputs in a paired table. Explain whether the short run demonstrates mechanics, behavior change, or reliable quality improvement.

## Optional QLoRA extension

On Linux/CUDA, repeat rank 8 with Qwen2.5-1.5B-Instruct. Pass `BitsAndBytesConfig` during `from_pretrained`, call `prepare_model_for_kbit_training`, and report base-layer dtypes. This extension is optional.

## Hints

<details><summary>Dataset format</summary>Use conversational `prompt` and `completion` columns. Completion-only loss is the default for prompt-completion data, but set it explicitly in `SFTConfig`.</details>

<details><summary>One-variable comparison</summary>Write both configurations to JSON and compare them. Aside from output directory and rank-dependent alpha if explicitly justified, the dictionaries should match.</details>

<details><summary>Reload test</summary>Construct a fresh `AutoModelForCausalLM` at the pinned revision, then call `PeftModel.from_pretrained`.</details>

## Suggested workflow

1. Freeze the shared data, decoding, and training settings in a configuration record.
2. Build rank-8 and rank-16 PEFT configurations and verify trainable parameters.
3. Train both adapters while changing only the requested rank variable.
4. Save and reload each adapter into a fresh pinned base model.
5. Compare paired outputs and resources, then write a calibrated conclusion.

## What to submit

Submit the notebook, two adapter directories, configuration JSON, trainable-parameter audit, loss table/plot, paired outputs, resource measurements, and limitations.

## Rubric (10 points)

- Reproducible real-data configuration (2)
- Trainable-parameter and objective checks (2)
- Controlled rank experiment and measurements (3)
- Save/reload evidence, paired outputs, and cautious conclusion (3)
