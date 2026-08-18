# Homework: Complete and Defend a Tool Loop

## Goal

Complete a model → validation → tool → result → model loop using the real USGS earthquake snapshot supplied with the course. Then demonstrate that three different failures are stopped or reported at the correct boundary. Expected working time: **60–75 minutes**.

## Before you begin

- Connect to the Clemson network or CUVPN.
- Complete `notebooks/00_setup_and_diagnostics.ipynb` and Lecture 02.
- Use `qwen3-30b-a3b-instruct-fp8`.
- Use `data/usgs_earthquakes_2025_01.jsonl`; do not invent earthquake records.
- Review `earthquake_lookup`, the tool schema, and the application `tool_map`.
- Do not submit API keys or complete private request headers.

Create `02_tools_agents_and_safety_homework_submission.ipynb`.

## Part A — Inspect the real tool contract

1. Find event `us6000pjqz` in the USGS JSONL file.
2. Record its `place`, `magnitude`, `depth_km`, and `status` directly from the file.
3. Explain the difference between:
   - a tool schema;
   - a Python implementation;
   - an application allowlist.
4. Confirm that the published schema accepts exactly one string argument named `event_id` and rejects additional properties.

## Part B — Complete one accepted tool loop

Use this request:

> Use the earthquake lookup tool to report the place, magnitude, depth, and review status for USGS event us6000pjqz. Do not guess missing fields.

1. Send the request with the `earthquake_lookup` tool schema.
2. Capture the model-proposed tool name and raw argument string.
3. Before executing anything, check all of the following in application code:
   - the tool name exists in `tool_map`;
   - the arguments parse as a JSON object;
   - the argument keys are exactly `{"event_id"}`;
   - `event_id` is a non-empty string;
   - the value contains only a reasonable USGS-style identifier, such as letters and digits.
4. Execute `earthquake_lookup` only after all checks pass.
5. Limit the tool-result text before adding it to the model context.
6. Append the assistant tool proposal and the `tool` result message to the conversation.
7. Ask the model for the final answer.
8. Verify the final answer against the real JSONL row. Identify every field that is supported by the tool result.

Create a trace table:

| stage | input summary | decision | output summary |
|---|---|---|---|
| model proposal |  | proposed only |  |
| schema/domain validation |  | accept/reject |  |
| tool execution |  | success/error |  |
| final response |  | supported/unsupported |  |

## Part C — Test three distinct failure boundaries

Run these cases without weakening your validator:

1. **Unknown tool:** a proposed tool named `delete_earthquake_record`.
2. **Extra argument:** `earthquake_lookup` with `{"event_id": "us6000pjqz", "path": "/etc/passwd"}`.
3. **Valid shape, missing record:** `earthquake_lookup` with `{"event_id": "notarealusgsid"}`.

For each case, record:

- the earliest boundary that should handle it;
- the observed exception or rejection category;
- whether the tool implementation ran;
- what message, if any, is safe to return to the model or user.

Explain why these are three different failures: allowlist failure, argument-validation failure, and tool/data failure.

## Optional extension — MCP

Call the local read-only earthquake MCP server for `us6000pjqz` and the missing ID. Compare MCP discovery and result envelopes with the direct Python tool loop. Discovery does not grant authorization.

This extension is not required for full credit.

## Hints

<details>
<summary>Hint 1 — A proposal is not an action</summary>

The first model response may contain `tool_calls`, but no Python function has run. Execution begins only when application code resolves an allowed name and invokes the mapped callable.

</details>

<details>
<summary>Hint 2 — Reject extra keys</summary>

After `json.loads`, compare `set(arguments)` with `{"event_id"}`. Checking only that `event_id` exists would still allow an attacker-controlled `path` argument.

</details>

<details>
<summary>Hint 3 — Keep errors at the correct layer</summary>

An unknown tool should fail before argument parsing. An extra key should fail before execution. A well-formed but unknown event ID reaches the read-only tool and should return a controlled not-found error.

</details>

<details>
<summary>Hint 4 — Bound tool output</summary>

The classroom row is small, but write the code as if a tool could return thousands of lines. Truncate or summarize before returning untrusted content to the model.

</details>

## What to submit

Submit the completed notebook containing:

- the real USGS row fields used for verification;
- your validation function;
- one complete accepted tool-loop trace;
- a three-row failure table;
- the final supported answer;
- one residual risk that remains after validation;
- no credentials or sensitive paths beyond the deliberately rejected fixture.

## Rubric (10 points)

- **Tool contract and real data (2):** schema, implementation, allowlist, and USGS reference are distinguished correctly.
- **Complete accepted loop (3):** proposal, validation, bounded execution, tool message, and final response are all present.
- **Failure-boundary tests (3):** all three cases are handled at the correct layer without unauthorized execution.
- **Evidence and trace quality (1):** decisions and outputs are recorded clearly and checked against the USGS row.
- **Residual risk and safe submission (1):** one meaningful risk is discussed and no credential is exposed.
