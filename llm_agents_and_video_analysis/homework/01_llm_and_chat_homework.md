# Homework: Decoding Controls and Conversation State

## Goal

Use live Clemson RCD models to answer two concrete questions:

1. What changes when we vary `temperature` while holding the request fixed?
2. What happens to a conversation constraint when the application does not resend the earlier message that introduced it?

This is a controlled experiment, not a contest to find the most impressive answer. Expected working time: **60–75 minutes**.

## Before you begin

- Connect to the Clemson network or CUVPN.
- Complete `notebooks/00_setup_and_diagnostics.ipynb`.
- Review the `run_chat()` and `message_content()` examples in Lecture 01.
- Use `qwen3-30b-a3b-instruct-fp8` for the required work.
- Do not place an API key in a notebook cell, output, screenshot, or submitted file.

Create a new notebook named `01_llm_and_chat_homework_submission.ipynb`. Record the model ID, `temperature`, `max_tokens`, prompt, and number of repetitions. Do not record the API key.

## Part A — Change one decoding control

Use this exact request for every required run:

> Explain why the daytime sky appears blue. Write exactly three bullet points for a high-school student. Each bullet must contain no more than 25 words.

1. Build one `messages` list containing the request.
2. Set `max_tokens=220`.
3. Generate three responses with `temperature=0.1`.
4. Generate three responses with `temperature=0.8`.
5. Do not change the model, messages, token budget, or any other setting between the two groups.
6. Store each final answer using `message_content()`.
7. Create a six-row results table with these columns:

| temperature | repetition | three_bullets | length_limit_met | wording_notes | factual_concern |
|---|---:|---|---|---|---|

The final two columns require your judgment. A fluent answer is not automatically correct.

Answer these questions in complete sentences:

- Which setting produced more wording variation? Cite at least two outputs.
- Did either setting violate the requested format?
- Why does a higher temperature not mean that the answer is more knowledgeable or more truthful?
- Why can repeated outputs still look similar even when sampling is enabled?

## Part B — Test application-owned conversation state

1. Send this first user message at `temperature=0.1` and `max_tokens=220`:

   > For every reply in this conversation, use exactly three bullet points written for a high-school student. First, explain why the daytime sky appears blue.

2. Save the assistant response.
3. Ask this follow-up question in two conditions:

   > Now explain why sunsets often appear red.

   - **Full-history condition:** send the first user message, first assistant response, and follow-up question.
   - **Dropped-history condition:** send only the follow-up question.

4. Record whether each response follows the original three-bullet and audience constraints.
5. Explain what the result shows about where conversation state is stored.
6. State one limitation of this experiment. For example, a model may happen to produce three bullets even when the earlier constraint was omitted.

## Optional extension — Thinking model

Repeat one request with `qwen3.5-9b` and `enable_thinking=True`. Use at least `max_tokens=1200` so that reasoning does not consume the entire output budget. Extract the final answer with `message_content()`; do not assume `message.content` is always an ordinary string.

This extension is not required for full credit.

## Hints

<details>
<summary>Hint 1 — Minimal call pattern</summary>

Use `message_content(run_chat(messages, CHAT_MODEL, client=client, max_tokens=220, temperature=value))`. Keep `messages` unchanged during Part A.

</details>

<details>
<summary>Hint 2 — Count the requested structure</summary>

Inspect the rendered response rather than relying only on string length. A numbered list is not literally the requested bullet-list format, and a bullet can exceed 25 words.

</details>

<details>
<summary>Hint 3 — Build the full-history request</summary>

The application should create a list with roles in this order: `user`, `assistant`, `user`. The dropped-history request contains only the last `user` message.

</details>

<details>
<summary>Hint 4 — Interpret variation cautiously</summary>

Temperature rescales token probabilities. It does not verify facts, guarantee creativity, or guarantee that two runs will differ.

</details>

## What to submit

Submit the completed notebook containing:

- a redacted configuration block;
- the six Part A responses and results table;
- both Part B responses and the constraint comparison;
- answers to all interpretation questions;
- one experimental limitation;
- no credentials or hidden reasoning text.

## Rubric (10 points)

- **Controlled temperature experiment (3):** six required runs; only temperature changes; settings are recorded.
- **Conversation-state experiment (3):** full and dropped histories are constructed correctly and compared.
- **Evidence and interpretation (2):** tables cite actual outputs and conclusions do not overstate what temperature or one trial proves.
- **Reproducibility and safe submission (1):** model, prompts, settings, and environment are documented without secrets.
- **Limitation and communication (1):** one meaningful limitation is stated clearly.
