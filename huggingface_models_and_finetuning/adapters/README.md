# Fine-tuning adapters

`classroom_smol_lora/` is a standard PEFT LoRA adapter produced by the real training path in Lecture 03. It uses the pinned SmolLM2-135M-Instruct base and authentic packaged OpenAssistant rows. The artifact is intentionally trained for only two optimizer steps: it supports classroom save/reload/merge exercises but does not claim a quality improvement.

`manifest.json` records the immutable base revision, data-manifest hash, software versions, configuration, observed training loss, and checksums. Extended Qwen2.5/QLoRA runs should be stored in a separate directory and accompanied by the same lineage information.
