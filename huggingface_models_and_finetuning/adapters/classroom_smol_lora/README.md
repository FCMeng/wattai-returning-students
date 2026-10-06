---
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
  - lora
  - sft
  - transformers
  - trl
---

# WattAI classroom LoRA systems-check adapter

This adapter accompanies the standalone Hugging Face and fine-tuning course. It was trained with PEFT on 16 human-authored conversation branches from the packaged `OpenAssistant/oasst1` classroom snapshot.

## Intended use

Use this artifact to practice adapter loading, parameter inspection, deterministic generation, lineage checks, and `merge_and_unload()`. It is not intended for deployment or as evidence that two optimizer steps improve instruction following.

## Base model

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- Upstream license: Apache-2.0

## Training configuration

- Method: completion-only supervised fine-tuning with LoRA
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Rank / alpha / dropout: 8 / 16 / 0.05
- Optimizer steps: 2
- Microbatch / accumulation: 1 / 1
- Maximum sequence length: 256
- Learning rate: `1e-4`
- Seed: 2026
- Software: PyTorch 2.8.0, Transformers 5.15.0, TRL 1.10.0, PEFT 0.20.0
- Observed training loss: 1.8067488074302673

## Verification

The adapter was loaded into a fresh pinned base with `is_trainable=False`. Greedy generation completed, and pre/post-merge generation matched on the verification prompt. Floating-point logits are compared with reported error rather than assumed to be bit-identical.

See `../manifest.json` for data lineage and checksums.
