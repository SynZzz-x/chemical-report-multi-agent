# Interview Demo Script

## 30-second overview

This is a LangGraph multi-agent system for data-analysis report generation. The core workflow is Intake, Planner, Worker, Verifier, Summarizer, and Exit. I focused on making the agent auditable: every task is planned, executed, checked, and only then merged into the final report.

## 90-second walkthrough

1. Start Streamlit and compile the workflow.
2. Upload `sample_energy_metrics.csv`.
3. Paste the demo prompt.
4. Show the Planner output: the system turns the request into tasks and maps tasks to resources.
5. Show Worker progress: each task produces structured output, charts, tables, or text.
6. Show Verifier: manual mode lets the user approve, request rework, or trigger replanning.
7. Show final report download buttons.

## Technical points to mention

- LangGraph gives explicit state transitions and checkpoint-friendly routing.
- Worker tools are isolated behind task-oriented interfaces.
- Verification separates generation from acceptance, reducing silent failures.
- Configuration was hardened so credentials come only from environment variables.
- Tests cover deterministic helper logic and scan for hardcoded secrets.
