# 报告准入与确定性汇总 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 阻止未验收章节进入正式报告，并以确定性拼装替代 Summarizer 的逐章 LLM 重写。

**Architecture:** 新增纯函数模块集中维护章节状态、报告状态与可交付章节筛选。Recovery 只负责写入状态和处理显式用户选择；Summarizer 根据推导出的报告状态决定阻塞、正式交付或带缺口草稿，并原样组装已获准章节。

**Tech Stack:** Python 3、LangGraph State、pytest、Markdown/PDF/DOCX 现有渲染工具。

## Global Constraints

- `section_status` 是报告准入的唯一状态来源，旧 `results` 不可自动视为已通过。
- `ACCEPT_WITH_WARNING` 不是用户接受，不得自动提交或推进。
- `BLOCKED` 不生成正式 PDF/DOCX。
- Summarizer 不调用 LLM，不改写已验证正文或证据编号。
- 本轮不扩展 Planner Task Contract。

---

### Task 1: 章节与报告状态纯函数

**Files:**
- Create: `src/report_acceptance.py`
- Modify: `src/state.py`
- Test: `tests/test_report_acceptance.py`

**Interfaces:**
- Produces: `record_section_status(state, status, accepted_by, issues=None) -> dict`
- Produces: `derive_report_status(tasks, section_status) -> str`
- Produces: `eligible_task_ids(tasks, section_status, report_status) -> list[str]`

- [ ] 写失败测试，覆盖正式报告、带缺口草稿、自动警告阻塞、缺失状态阻塞和任务顺序。
- [ ] 运行 `pytest tests/test_report_acceptance.py -q`，确认因模块或接口不存在而失败。
- [ ] 实现最小纯函数并扩展 State 字段。
- [ ] 再次运行测试并确认通过。

### Task 2: Recovery 准入状态流转

**Files:**
- Modify: `src/recovery/policy.py`
- Modify: `src/nodes/recovery.py`
- Modify: `src/nodes/verifier_manual.py`
- Modify: `src/control_messages.py`
- Test: `tests/test_recovery_policy.py`
- Test: `tests/test_recovery_graph.py`

**Interfaces:**
- Consumes: Task 1 的状态纯函数。
- Produces: PASS 写入 `VERIFIED_PASS`；重试耗尽写入 `ACCEPT_WITH_WARNING` 并请求用户；显式接受写入用户接受状态并提交。

- [ ] 修改测试，要求重试耗尽不提交、进入 `NEEDS_USER_INPUT`，并提供显式接受草稿动作。
- [ ] 运行 recovery 测试，确认旧行为导致断言失败。
- [ ] 修改 policy 和恢复节点的最小状态流转。
- [ ] 更新人工审核和控制消息，使显式接受可审计且可恢复。
- [ ] 运行 recovery 测试并确认通过。

### Task 3: 确定性 Summarizer

**Files:**
- Modify: `src/nodes/summarizer_v2.py`
- Modify: `src/evidence/reporting.py`
- Test: `tests/test_summarizer_deterministic.py`

**Interfaces:**
- Consumes: `section_status` 与 `derive_report_status()`。
- Produces: `READY_FOR_FINAL` 正式附件、`DRAFT_WITH_GAPS` 草稿附件、`BLOCKED` 阻塞结果。

- [ ] 写失败测试，覆盖阻塞不渲染、正式/草稿命名、风险说明、任务顺序、安全去重标题和无 LLM 调用。
- [ ] 运行 Summarizer 测试，确认现有实现不满足准入和确定性要求。
- [ ] 删除逐章 LLM 与 LLM 评价路径，改为确定性拼装。
- [ ] 统一 citations 附录并只返回实际生成成功的附件。
- [ ] 运行 Summarizer 测试并确认通过。

### Task 4: 回归、审查与交付

**Files:**
- Modify: only files required by failed regression tests.

**Interfaces:**
- Consumes: Tasks 1-3 全部实现。
- Produces: 可推送的 `codex/sqlite-checkpoint-store` 提交。

- [ ] 运行 recovery、graph state、summarizer 定向测试。
- [ ] 运行完整 `pytest -q` 与 `git diff --check`。
- [ ] 审查 diff，检查自动提交失败结果、Summarizer LLM 调用和失败占位符已消失。
- [ ] 提交并推送 `codex/sqlite-checkpoint-store`。
