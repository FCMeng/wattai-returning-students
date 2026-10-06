# Hugging Face Models and Fine-Tuning

This standalone asynchronous course contains four self-paced tutorials. Each tutorial usually takes 60–90 minutes, excluding model downloads and optional large-model experiments. Students begin with a real Hugging Face model, turn authentic OpenAssistant conversations into a supervised objective, train and reload a PEFT LoRA adapter, and finish with paired evaluation and packaging.

## What students need

Students should know ordinary Python, lists and dictionaries, and basic algebra. Prior LLM or Hugging Face experience is not required. The notebooks recap token IDs, logits, next-token loss, train/evaluation separation, and the role of a chat template when each idea is first used.

Initial setup requires internet access to install packages and download model weights. Cached model files can be reused afterward. The reduced path uses a real 135M-parameter SmolLM2 model on CPU or GPU; the Qwen2.5-1.5B and QLoRA extensions require a suitable Linux/CUDA environment.

The course uses one tested package set: PyTorch 2.8.0 with torchvision 0.23.0, Transformers 5.17.0, PEFT 0.21.2, and TRL 1.14.1. Transformers 5.17.0 is required because the Qwen3.5 lesson imports `Qwen3_5ForConditionalGeneration`; installing it beside an older PEFT or an incompatible torchvision can cause import errors even before a model is loaded.

For the recommended experience, use **Palmetto 2 with a CUDA GPU**. Lectures 01–02 can run on CPU, but model generation is slower; LoRA and especially QLoRA are better suited to the GPU environment.

## Environment setup

Open a terminal in this course folder and create the course environment before opening the setup notebook:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name wattai-hf-finetuning --display-name "WattAI: Hugging Face Fine-Tuning"
```

Restart JupyterLab and select **WattAI: Hugging Face Fine-Tuning** as the kernel. On Palmetto 2, follow the current cluster guidance if its recommended CUDA-enabled PyTorch installation differs from `requirements.txt`.

Upgrade the stack as a unit rather than installing a newer Transformers package into an older environment. If the setup notebook reports mixed versions, creating a fresh environment is the most reliable repair; after any package change, restart the notebook kernel before importing Transformers, PEFT, or TRL again.

## Start here

1. Open `notebooks/00_setup_and_diagnostics.ipynb`.
2. Confirm that the environment, Palmetto 2 GPU, classroom data, tokenizer, and model configuration pass the checks.
3. Run Lectures 01–04 in order if this is your first fine-tuning course. Lecture 01 downloads the 135M model weights on its first run.

Work cell by cell rather than using **Run All** in Lecture 01. Its large-model switches are enabled so an individual experiment runs without code edits, but Qwen3.5, Qwen3-VL, and Qwen3-Omni have different GPU requirements. Run only the section supported by your allocation, then continue with the text and reflection sections.

Each lecture is independently readable, but Lecture 03 writes the adapter used by Lecture 04. A small verified classroom adapter is also supplied in `adapters/classroom_smol_lora/` so Lecture 04 can be taught directly.

## Notebook sequence

| Notebook | Main question | Reduced path |
|---|---|---|
| `01_huggingface_ecosystem.ipynb` | Should this problem be solved by prompting, evidence, or weight updates? | Run SmolLM2, compare Qwen3.5 thinking/direct modes, and inspect image/video model contracts. |
| `02_finetuning_data_and_objective.ipynb` | Which exact tokens should contribute to SFT loss? | Audit 2,000/250 OpenAssistant rows and render real chat templates. |
| `03_supervised_finetuning.ipynb` | How can a low-rank update adapt a frozen model? | Run two real LoRA optimizer steps, save, and reload a PEFT adapter. |
| `04_peft_qlora_and_evaluation.ipynb` | How do we compare and package the adapter without overstating results? | Generate paired records, review failures, reload, and merge. |

## Data provenance

The packaged classroom data comes from `OpenAssistant/oasst1` revision `694bedf` under Apache-2.0. It contains 2,000 training conversations and 250 held-out conversations. Splits are formed by original message-tree ID so branches sharing ancestors cannot cross the boundary.

`data/manifests/classroom.json` is the dataset's packing list, not an additional training dataset. It records provenance, split policy, row counts, file sizes, and SHA-256 fingerprints so students can confirm that they are using the intended files. Paths inside it are relative to this standalone course folder.

The course does not substitute invented conversations for experimental results. Small hand calculations may be used to explain a formula, but all dataset audits, tokenization, training, and evaluation paths operate on the packaged OpenAssistant records.

## Models and artifacts

- Classroom instruction model: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Classroom base comparison: `HuggingFaceTB/SmolLM2-135M`, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- CUDA extension: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Thinking/direct and native-multimodal lab: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Vision-language lab: `Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- Large-GPU video extension: `Qwen/Qwen3-Omni-30B-A3B-Instruct`, revision `26291f793822fb6be9555850f06dfe95f2d7e695`.
- Large-GPU thinking extension: `Qwen/Qwen3-Omni-30B-A3B-Thinking`, revision `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b`.

The listed upstream model cards state Apache-2.0. The Qwen3.5 and Qwen3-VL weights are large optional downloads. Qwen3-Omni requires substantially more GPU memory and is not intended for an ordinary 24 GB allocation. Its Python preprocessing package is included in `requirements.txt`, while `ffmpeg` remains a system dependency. Lecture 01 defaults to PyTorch SDPA; FlashAttention 2 is recommended but has a separate opt-in switch because its compiled package must match the local CUDA and PyTorch environment. The supplied `classroom_smol_lora` adapter is a systems-check artifact trained for two optimizer steps on 16 packaged OpenAssistant rows. It proves that PEFT training, save, reload, generation, and merge workflows execute; it is **not** evidence of meaningful quality improvement.

## Homework and solutions

Each lecture has a specific assignment in `homework/` with fixed models/data, required steps, hints, submission items, and a tailored ten-point rubric. Instructor solutions are stored separately in `solutions/`; omit that directory from the student distribution.

## Standalone boundary

Student notebooks import only `course_helpers.py` from this folder. The first important helper is shown in full before it is imported, so the module removes repetition without hiding the behavior being taught. No notebook requires the repository-level `shared/` package or output from another course.
