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
from tests.test_offline_pipeline_benchmark import (
    collect_shared_e3_catalog_metrics,
    collect_verifier_pass_metrics,
)


def capture_request(
    env: dict[str, str],
    *,
    bound_kwargs: dict[str, object] | None = None,
    apply_completion_budget: bool = True,
    with_listener: bool = False,
    with_types: bool = False,
) -> dict[str, object]:
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
        runnable = model.bind(**bound_kwargs) if bound_kwargs else model
        listener_events: list[str] = []
        if with_listener:
            runnable = runnable.with_listeners(
                on_start=lambda *_args: listener_events.append("start")
            )
        if with_types:
            runnable = runnable.with_types(str, str)
        captured["config_factories_before"] = len(
            getattr(runnable, "config_factories", [])
        )
        captured["input_type_before"] = getattr(
            runnable, "custom_input_type", None
        ).__name__ if getattr(runnable, "custom_input_type", None) else None
        captured["output_type_before"] = getattr(
            runnable, "custom_output_type", None
        ).__name__ if getattr(runnable, "custom_output_type", None) else None
        if apply_completion_budget:
            runnable, _ = with_completion_budget(runnable, "assessment")
        captured["config_factories_after"] = len(
            getattr(runnable, "config_factories", [])
        )
        captured["input_type_after"] = getattr(
            runnable, "custom_input_type", None
        ).__name__ if getattr(runnable, "custom_input_type", None) else None
        captured["output_type_after"] = getattr(
            runnable, "custom_output_type", None
        ).__name__ if getattr(runnable, "custom_output_type", None) else None
        runnable.invoke([HumanMessage(content="offline verifier probe")])
        captured["listener_events"] = listener_events
    return captured


def _request_fields(
    payload: dict[str, object], *, include_bound_fields: bool = False
) -> dict[str, object]:
    fields = {
        "model": payload.get("model"),
        "max_tokens": payload.get("max_tokens"),
        "max_completion_tokens": payload.get("max_completion_tokens"),
        "reasoning_effort": payload.get("reasoning_effort"),
        "thinking_present": "thinking" in payload,
    }
    if include_bound_fields:
        fields.update(
            {
                "bound_flag": payload.get("bound_flag"),
                "tool_choice": payload.get("tool_choice"),
                "tools": payload.get("tools"),
                "listener_events": payload.get("listener_events"),
                "config_factories_before": payload.get("config_factories_before"),
                "config_factories_after": payload.get("config_factories_after"),
                "input_type_before": payload.get("input_type_before"),
                "input_type_after": payload.get("input_type_after"),
                "output_type_before": payload.get("output_type_before"),
                "output_type_after": payload.get("output_type_after"),
            }
        )
    return fields


def run_verifier_control_probe(
    env: dict[str, str],
    *,
    bound_kwargs: dict[str, object] | None = None,
    apply_completion_budget: bool = True,
    with_listener: bool = False,
    with_types: bool = False,
) -> dict[str, object]:
    """Run the installed wrapper in a child process outside pytest's stubs."""

    command = [sys.executable, str(Path(__file__).resolve()), "--capture"]
    if bound_kwargs:
        command.extend(["--bound-json", json.dumps(bound_kwargs)])
    if not apply_completion_budget:
        command.append("--skip-completion-budget")
    if with_listener:
        command.append("--with-listener")
    if with_types:
        command.append("--with-types")
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


def _optimized_artifact() -> dict[str, object]:
    first = run_verifier_control_probe({})
    second = run_verifier_control_probe({})
    first_components = collect_verifier_pass_metrics()
    second_components = collect_verifier_pass_metrics()
    first_catalog = collect_shared_e3_catalog_metrics()
    second_catalog = collect_shared_e3_catalog_metrics()
    assert first == second
    assert first_components == second_components
    assert first_catalog == second_catalog
    assert first["max_tokens"] == 1600
    assert first["max_completion_tokens"] is None
    assert first["reasoning_effort"] is None
    assert first["thinking_present"] is False
    return {
        "verifier_prompt_components": first_components,
        "semantic_catalog": first_catalog,
        "request_mapping": first,
        "provider_tokens": None,
        "online_latency_seconds": None,
        "online_latency_remeasured": False,
        "requires_real_run": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--bound-json")
    parser.add_argument("--skip-completion-budget", action="store_true")
    parser.add_argument("--with-listener", action="store_true")
    parser.add_argument("--with-types", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.capture:
        bound_kwargs = json.loads(args.bound_json) if args.bound_json else None
        print(
            "__OFFLINE_VERIFIER_CONTROL_CAPTURE__"
            + json.dumps(
                _request_fields(
                    capture_request(
                        {},
                        bound_kwargs=bound_kwargs,
                        apply_completion_budget=not args.skip_completion_budget,
                        with_listener=args.with_listener,
                        with_types=args.with_types,
                    ),
                    include_bound_fields=(
                        bound_kwargs is not None or args.with_listener or args.with_types
                    ),
                ),
                sort_keys=True,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required unless --capture is used")

    artifact = json.loads(args.output.read_text(encoding="utf-8"))
    artifact["optimized"] = _optimized_artifact()
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["optimized"]["request_mapping"], sort_keys=True))


if __name__ == "__main__":
    main()
