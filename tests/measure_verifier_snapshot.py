"""Measure the fixed PASS Verifier fixture against an extracted git snapshot."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_support import (
    BenchmarkRecorder,
    PASS_VERIFIER_RESPONSE,
    VERIFIER_PASS_STATE,
    measure_serialized_messages,
    serialize_emitted_response,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.snapshot.resolve()))

    from src.nodes import verifier as verifier_module

    recorder = BenchmarkRecorder(
        SimpleNamespace(
            content=json.dumps(PASS_VERIFIER_RESPONSE, ensure_ascii=False)
        )
    )

    def fake_invoke(model, value, **kwargs):
        return model.invoke(value)

    with patch.object(verifier_module, "get_llm", lambda *args, **kwargs: recorder), patch.object(
        verifier_module, "invoke_llm", fake_invoke
    ):
        update = verifier_module.verifier(
            deepcopy(VERIFIER_PASS_STATE),
            {"configurable": {"use_llm": True}},
        )

    result = {
        "serialized_prompt_chars": measure_serialized_messages(recorder.calls)[
            "serialized_prompt_chars"
        ],
        "mock_completion_chars": len(serialize_emitted_response(recorder.response)),
        "semantic_llm_calls": len(recorder.calls),
        "assessment_status": update["assessment"]["status"],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
