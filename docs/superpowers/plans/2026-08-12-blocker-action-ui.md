# Blocker Action UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户直接在阻塞处理区完成授权、接受缺口、调整要求或上传资料，并通过现有 LangGraph resume 链继续工作流。

**Architecture:** 在 `src/control_messages.py` 增加无 UI 依赖的动作描述与提交校验函数，`app.py` 只负责按描述渲染 Streamlit 控件。专用控件与普通聊天输入统一转换为一个提交对象，再复用当前文件保存、消息持久化、`Command(resume=resume_payload)` 和流式执行路径。

**Tech Stack:** Python 3.13、Streamlit、LangGraph、pytest

## Global Constraints

- 只展示后端 `accepted_choices` 允许的动作。
- `UPLOAD_RESOURCES` 必须包含至少一个文件，`ADJUST_REQUIREMENT` 必须包含非空文本。
- `AUTHORIZE_WEB` 与 `ACCEPT_EVIDENCE_GAP` 可一键提交。
- 普通聊天输入行为不变。

---

### Task 1: 阻塞动作提交契约

**Files:**
- Modify: `src/control_messages.py`
- Test: `tests/test_recovery_compatibility.py`

**Interfaces:**
- Consumes: blocker action code、用户文本、上传文件数量。
- Produces: `blocker_action_spec(action: str) -> dict[str, Any]` 与 `validate_blocker_submission(action: str, text: str, document_count: int) -> str | None`。

- [ ] **Step 1: Write the failing tests**

```python
def test_blocker_action_specs_define_direct_submission_requirements():
    from src.control_messages import blocker_action_spec

    assert blocker_action_spec("AUTHORIZE_WEB")["default_text"] == "已授权公开网络检索，请继续。"
    assert blocker_action_spec("ACCEPT_EVIDENCE_GAP")["default_text"] == "接受现有证据及缺口报告，请继续。"
    assert blocker_action_spec("ADJUST_REQUIREMENT")["requires_text"] is True
    assert blocker_action_spec("UPLOAD_RESOURCES")["requires_documents"] is True


def test_blocker_submission_validation_rejects_missing_required_input():
    from src.control_messages import validate_blocker_submission

    assert validate_blocker_submission("ADJUST_REQUIREMENT", "", 0)
    assert validate_blocker_submission("UPLOAD_RESOURCES", "", 0)
    assert validate_blocker_submission("AUTHORIZE_WEB", "", 0) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_recovery_compatibility.py -k 'blocker_action_specs or blocker_submission_validation' -q`

Expected: FAIL because the two helpers do not exist.

- [ ] **Step 3: Implement the minimal pure helpers**

```python
BLOCKER_ACTION_SPECS = {
    "AUTHORIZE_WEB": {
        "label": "授权公开网络检索",
        "button_label": "授权并继续",
        "default_text": "已授权公开网络检索，请继续。",
        "requires_text": False,
        "requires_documents": False,
    },
    "ACCEPT_EVIDENCE_GAP": {
        "label": "接受现有证据及缺口报告",
        "button_label": "接受并继续",
        "default_text": "接受现有证据及缺口报告，请继续。",
        "requires_text": False,
        "requires_documents": False,
    },
    "ADJUST_REQUIREMENT": {
        "label": "调整任务要求",
        "button_label": "提交新要求",
        "default_text": "",
        "requires_text": True,
        "requires_documents": False,
    },
    "UPLOAD_RESOURCES": {
        "label": "上传补充资料",
        "button_label": "上传并继续",
        "default_text": "我已上传补充资料，请结合附件继续处理当前任务。",
        "requires_text": False,
        "requires_documents": True,
    },
}


def blocker_action_spec(action: str) -> dict[str, Any]:
    return dict(BLOCKER_ACTION_SPECS.get(str(action).strip(), {}))


def validate_blocker_submission(action: str, text: str, document_count: int) -> str | None:
    spec = blocker_action_spec(action)
    if spec.get("requires_text") and not str(text).strip():
        return "请输入调整后的任务要求。"
    if spec.get("requires_documents") and document_count <= 0:
        return "请先上传补充资料。"
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_recovery_compatibility.py -k 'blocker_action_specs or blocker_submission_validation' -q`

Expected: PASS.

### Task 2: Streamlit 动作专用控件与统一提交

**Files:**
- Modify: `app.py`
- Test: `tests/test_recovery_compatibility.py`

**Interfaces:**
- Consumes: `blocker_choices()`、`blocker_action_spec()`、`validate_blocker_submission()`。
- Produces: `_render_pending_resume_submission() -> dict[str, Any] | None`，返回 `action/text/files`；普通聊天提交保持同样结构。

- [ ] **Step 1: Write a failing integration test**

```python
def test_streamlit_uses_action_specific_blocker_submission_controls():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "_render_pending_resume_submission" in source
    assert 'st.file_uploader(' in source
    assert 'st.text_area(' in source
    assert 'button_label' in source
    assert "pending_resume_action = _render_pending_resume_action()" not in source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_recovery_compatibility.py::test_streamlit_uses_action_specific_blocker_submission_controls -q`

Expected: FAIL because the current UI only renders a selectbox and relies on `st.chat_input`.

- [ ] **Step 3: Implement the dedicated blocker controls**

```python
def _render_pending_resume_submission() -> dict[str, Any] | None:
    payload = st.session_state.get("pending_interrupt")
    choices = blocker_choices(payload)
    if not choices:
        return None
    action = st.selectbox(
        "选择当前阻塞的处理方式",
        options=[""] + choices,
        format_func=lambda value: (
            "请选择……" if not value else blocker_action_spec(value)["label"]
        ),
        key="pending_blocker_action",
    )
    spec = blocker_action_spec(action)
    text = (
        st.text_area("输入调整后的任务要求", key="pending_blocker_text")
        if spec.get("requires_text")
        else spec.get("default_text", "")
    )
    files = (
        st.file_uploader(
            "上传补充资料",
            type=["csv", "pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="pending_blocker_files",
        )
        if spec.get("requires_documents")
        else []
    )
    if st.button(spec["button_label"], disabled=not action):
        error = validate_blocker_submission(action, text, len(files or []))
        if error:
            st.warning(error)
            return None
        return {"action": action, "text": text, "files": list(files or [])}
    return None
```

Move the shared submission work into one path that accepts either the dedicated blocker submission or `st.chat_input`. Disable ordinary chat input while a blocker is pending so the action cannot be omitted accidentally.

- [ ] **Step 4: Run related tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_recovery_compatibility.py -q`

Expected: PASS.

### Task 3: Regression verification

**Files:**
- Verify: `app.py`
- Verify: `src/control_messages.py`
- Verify: `tests/test_recovery_compatibility.py`

**Interfaces:**
- Consumes: completed implementation.
- Produces: verified source and test results.

- [ ] **Step 1: Run syntax and diff checks**

Run: `PYTHONPATH=. .venv/bin/python -m compileall -q app.py src tests`

Expected: exit code 0.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 2: Run project regression tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q --ignore=.worktrees`

Expected: all repository tests pass, excluding the known external worktree security-scan environment.

- [ ] **Step 3: Review and commit**

Run: `git diff -- app.py src/control_messages.py tests/test_recovery_compatibility.py`

Expected: only the approved blocker-interaction behavior changes.

Commit message: `feat: add direct blocker action controls`.
