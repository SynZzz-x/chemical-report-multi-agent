from pathlib import Path


def test_requirements_is_utf8_readable():
    text = Path("requirements.txt").read_text(encoding="utf-8")
    assert "langgraph==1.0.1" in text
    assert "\x00" not in text


def test_langgraph_checkpoint_version_is_compatible_with_langchain_core_1_0():
    lines = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "langgraph-checkpoint==3.0.1" in lines
