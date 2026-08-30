"""Capture the installed ChatOpenAI request shape without contacting a provider."""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI as RealChatOpenAI

from src.llm import get_llm, with_completion_budget
from tests.test_offline_pipeline_benchmark import collect_verifier_pass_metrics


def capture_request(env: dict[str, str]) -> dict[str, object]:
    captured: dict[str, object] = {}
    probe_env = {
        "DEEPSEEK_API_KEY": "offline-test-key",
        "DEEPSEEK_BASE_URL": "https://offline.invalid/v1",
        "DEEPSEEK_MODEL": "deepseek-v4-flash",
        **env,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": captured["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    with patch.dict(os.environ, probe_env, clear=False), patch(
        "src.llm.ChatOpenAI", functools.partial(RealChatOpenAI, http_client=client)
    ):
        model = get_llm({}, json_mode=True, purpose="assessment")
        runnable, _ = with_completion_budget(model, "assessment")
        runnable.invoke([HumanMessage(content="offline verifier probe")])
    return captured


def _request_fields(payload: dict[str, object]) -> dict[str, object]:
    return {
        "model": payload.get("model"),
        "max_tokens": payload.get("max_tokens"),
        "max_completion_tokens": payload.get("max_completion_tokens"),
        "reasoning_effort": payload.get("reasoning_effort"),
        "thinking_present": "thinking" in payload,
    }


def run_verifier_control_probe(env: dict[str, str]) -> dict[str, object]:
    """Run the installed wrapper in a child process outside pytest's stubs."""

    command = [sys.executable, str(Path(__file__).resolve()), "--capture"]
    safe_environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUTF8": "1",
        **env,
    }
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=safe_environment,
    )
    marker = "__OFFLINE_VERIFIER_CONTROL_CAPTURE__"
    line = next(
        line for line in reversed(result.stdout.splitlines()) if line.startswith(marker)
    )
    return json.loads(line.removeprefix(marker))


def _baseline_artifact() -> dict[str, object]:
    first = run_verifier_control_probe({})
    second = run_verifier_control_probe({})
    first_components = collect_verifier_pass_metrics()
    second_components = collect_verifier_pass_metrics()
    assert first == second
    assert first_components == second_components
    assert first["max_completion_tokens"] == 1600
    assert first["max_tokens"] is None
    assert first["reasoning_effort"] is None
    assert first["thinking_present"] is False
    return {
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "offline_only": True,
        "character_metrics_are_provider_tokens": False,
        "online_latency_remeasured": False,
        "requires_real_run": True,
        "verifier_prompt_components": first_components,
        "request_mapping": {
            "expected_provider_field": "max_tokens",
            "observed_budget_field_before_fix": "max_completion_tokens",
            "reasoning_effort_configured": False,
            "thinking_controlled": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.capture:
        print(
            "__OFFLINE_VERIFIER_CONTROL_CAPTURE__"
            + json.dumps(_request_fields(capture_request({})), sort_keys=True)
        )
        return
    if args.output is None:
        parser.error("--output is required unless --capture is used")

    artifact = _baseline_artifact()
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["request_mapping"], sort_keys=True))


if __name__ == "__main__":
    main()
