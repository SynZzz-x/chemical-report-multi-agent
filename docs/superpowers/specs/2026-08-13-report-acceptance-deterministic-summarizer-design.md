# 报告准入与确定性汇总设计

## 目标

让 Verifier 的验收结果真正约束最终交付：未通过或仅由系统自动降级的章节不得被静默包装成正式报告；已通过或由用户明确接受证据缺口的章节，以确定性方式拼装成正式报告或带缺口草稿。

## 范围

本轮只实现两个 P0：

1. 章节级验收状态与作业级报告状态；
2. 不调用 LLM 的确定性 Summarizer。

Summary/Conclusion 专用上下文、高风险工业控制建议校验、图形降级策略、Catalog 能力增强和通用 Markdown 表格解析器不在本轮范围内。

## 章节状态

`section_status` 以 `task_id` 为键，是章节是否有资格进入报告的唯一状态来源。每条记录包含：

```json
{
  "status": "VERIFIED_PASS",
  "accepted_by": "verifier",
  "issues": [],
  "plan_revision": 1,
  "task_revision": 1
}
```

状态语义：

- `VERIFIED_PASS`：Verifier 或人工审核明确通过，可进入正式报告；
- `USER_ACCEPTED_GAP`：用户明确接受证据缺口，只能进入带缺口草稿；
- `USER_ACCEPTED_WARNING`：用户明确接受非证据类内容缺陷，只能进入带缺口草稿；
- `ACCEPT_WITH_WARNING`：系统重试耗尽后的自动降级状态，不等于用户接受，不可提交或继续；
- `BLOCKED`：证据或内容仍未满足要求；
- `EXTERNAL_BLOCKER`：资源、权限或要求冲突等外部阻塞。

旧 checkpoint 若只有 `results` 而没有 `section_status`，按阻塞处理，不反向推断其已通过。

## 报告状态

`report_status` 由任务列表与 `section_status` 确定性推导：

- 全部任务均为 `VERIFIED_PASS`：`READY_FOR_FINAL`；
- 至少一个任务为 `USER_ACCEPTED_GAP` 或 `USER_ACCEPTED_WARNING`，且无阻塞状态：`DRAFT_WITH_GAPS`；
- 存在缺失状态、`ACCEPT_WITH_WARNING`、`BLOCKED` 或 `EXTERNAL_BLOCKER`：`BLOCKED`。

## 恢复与用户交互

- Verifier PASS：提交 `current_result`，记录 `VERIFIED_PASS`；
- 内容重试耗尽：记录 `ACCEPT_WITH_WARNING`，进入 `NEEDS_USER_INPUT`，不得提交当前结果；
- 证据恢复耗尽或外部阻塞：记录 `BLOCKED`/`EXTERNAL_BLOCKER`，进入 `NEEDS_USER_INPUT`；
- 用户可选择返工、调整要求、接受为带风险草稿或结束；
- 只有显式选择接受时，系统才提交当前结果并把状态改为 `USER_ACCEPTED_GAP` 或 `USER_ACCEPTED_WARNING`。

## 确定性 Summarizer

Summarizer 不调用大模型，不再次改写已验证正文。处理流程：

1. 推导 `report_status`；
2. `BLOCKED` 时停止生成 PDF/DOCX，返回具体阻塞章节和问题；
3. `READY_FOR_FINAL` 时只读取 `VERIFIED_PASS` 章节并生成 `report.*`；
4. `DRAFT_WITH_GAPS` 时读取已通过及用户接受章节，插入醒目的缺口说明，并生成 `report_draft_with_gaps.*`；
5. 按 Planner 任务顺序组装章节，保留原始正文、证据编号、正式表格和图形资产；
6. 仅当正文开头第一个 Markdown 标题与任务名高度一致时删除该重复标题；
7. 将真实 citations 去重后生成统一证据来源附录。

生成失败的文件不得出现在附件列表中，也不得写入 `(LLM Generation Failed)` 之类占位内容。

## 验证

- 单元测试覆盖状态推导的所有优先级；
- 恢复策略测试证明重试耗尽不再提交结果；
- 用户接受测试证明状态转换和提交只发生一次；
- Summarizer 测试证明 BLOCKED 不生成正式文件、草稿带风险说明、正式报告不调用 LLM、重复标题仅安全删除一次；
- 运行现有 recovery、graph state 和报告相关回归测试。
