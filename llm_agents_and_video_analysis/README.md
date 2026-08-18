# LLM Agents and Video Analysis

This is a standalone course for classes using Clemson RCD LLM access. Language-model vocabulary, structured messages, tool boundaries, and evaluation are introduced within the course, so it does not require the other WattAI courses. Every API exercise requires the Clemson network or CUVPN, an active allocation, and an RCD API key.

Three detailed, 60–90 minute Jupyter lectures for Clemson researchers and users of compatible local endpoints. The course defaults to locally hosted models through the RCD LLM Service and finishes with short-video analysis.

## Course files

- `notebooks/00_setup_and_diagnostics.ipynb` — access, API key, connectivity, and exact-model preflight
- `notebooks/01_llm_and_chat.ipynb` — next-token prediction, chat messages, and conversation state
- `notebooks/02_tools_agents_and_safety.ipynb` — tool calling, the agent loop, context, and safety
- `notebooks/03_video_analysis.ipynb` — video-capable model discovery, structured analysis, and evaluation
- `course_helpers.py` — shared API, validation, model-output, video, and tool-loop helpers
- `earthquake_mcp_server.py` — small local, read-only MCP server used by Lecture 2
- `assets/video_ground_truth.json` — human-reviewed reference annotations

Each notebook combines concise teaching notes, live executable exercises, prediction prompts, checkpoints, local SVG figures, and collapsible **Code walkthrough** sections. Important helper functions are first written and explained in the notebook; after students understand them, the maintained copies are imported from `course_helpers.py`.

## Student setup

1. Obtain an approved RCD LLM Service allocation and API key.
2. Connect to the Clemson network or CUVPN.
3. Start a Python 3.10+ Open OnDemand Jupyter session with 1 CPU, 4 GB memory, and no GPU.
4. Install the small course dependency set:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Open and complete `notebooks/00_setup_and_diagnostics.ipynb` before class.

The notebooks prompt securely for an API key when `RCD_LLM_API_KEY` is not set. They never save the key in notebook source or output.

Lecture 1 demonstrates two RLS endpoints with the same RCD key:

- `https://llm.rcd.clemson.edu/v1` for locally hosted Gemma and Qwen models;
- `https://llm.rcd.clemson.edu/openai/v1` for an eligible text-only request to an OpenAI-hosted model.

The OpenAI gateway has separate credits and policy checks. Current RLS policy
blocks image/file inputs through the gateway, so Lecture 3 sends video only to
the locally hosted Qwen3-Omni model.

Lecture 2 also contrasts the local stdio server with a public, read-only remote
MCP request to `https://mcp.deepwiki.com/mcp`. That example sends only the name
of a public GitHub repository and does not use the student's RCD or GitHub key.
Because it is an external live service, its availability and advertised schema
must be checked when the notebook runs.

## Instructor setup

The repository includes NASA Scientific Visualization Studio item 30628, “Trio of Hurricanes Over the Pacific Ocean,” with source metadata, checksum, and human-reviewed temporal annotations. See `assets/VIDEO_ASSET.md`.

Before class, run the setup notebook from the same Clemson/VPN environment the
students will use. It checks the exact fixed local model IDs. Rehearse Lecture
1's separately governed OpenAI text example after confirming gateway credits.

The student distribution contains ready-to-run notebooks; no notebook-generation scripts are required.

## Data and safety

Use only public, licensed, or otherwise approved videos. Do not submit confidential or restricted video without the required Clemson security approval. Model timestamps are estimates, not verified annotations.

## Documentation

- [RCD LLM Service](https://docs.rcd.clemson.edu/llm/)
- [Local Model API](https://docs.rcd.clemson.edu/llm/usage/api/)
- [Available Models](https://docs.rcd.clemson.edu/llm/models/)
- [Acceptable Use Guidelines](https://docs.rcd.clemson.edu/llm/acceptable_use/)
