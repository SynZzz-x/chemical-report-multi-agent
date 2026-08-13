# Verifier 路由与用户决策日志设计

## 目标

修复同一次审核同时包含确定性内容缺陷与 Verifier 协议故障时的错误路由，并让阻塞恢复后的用户选择进入服务端日志。

## 路由设计

`ASSESSMENT_CONTRACT_ERROR` 是 Verifier 健康信号，不参与普通内容 issue 的优先级竞争。路由按以下顺序仲裁：

1. 如果存在 `ASSESSMENT_CONTRACT_ERROR` 以外的 issue，只在其他 issue 中选择现有最高优先级类别。
2. 只有 assessment 中不存在任何可执行的内容、证据、计划或外部阻塞 issue 时，才按 `VERIFIER_FAILURE` 处理。
3. 因此 `ASSESSMENT_CONTRACT_ERROR + TOO_LONG` 路由为 `CONTENT_DEFECT → REWORK`；单独的 `ASSESSMENT_CONTRACT_ERROR` 仍按现有上限重试 Verifier，耗尽后进入 `NEEDS_USER_INPUT`。
4. `LLM_ERROR`、`LLM_NOT_ENABLED` 等真正的运行时/服务故障保持现有优先级，因为此时语义审核并未成功执行。

此修改只改变 assessment 的类别仲裁，不删除原始 issue，保留完整审计信息。

## 用户决策日志

`needs_user_input` 在解析并验证恢复动作后记录一条 INFO 日志，至少包含：

- 当前 `task_id`；
- 阻塞类别；
- 用户选择的标准化 choice；
- 最终工作流 action；
- 是否上传文件。

日志在动作确定后、状态返回前写入。它不记录用户自由文本或文件路径，避免把可能敏感的信息写进服务端日志。

`decision_policy` 同时记录 `source=system` 的自动路由日志。由此，终端中推进到下一任务之前若出现 `source=user`，代表用户明确接受或选择了恢复动作；出现 `source=system` 则代表策略自动路由。

## 范围边界

- 不修改 `covers_sections`、任务粒度或 Planner Task Contract。
- 不修改现有重试次数。
- 不改变用户明确选择 `ACCEPT_AS_DRAFT` 或 `ACCEPT_EVIDENCE_GAP` 后进入下一任务的行为。
- Planner 已有“无真实 CSV 不得规划定量分析/热力图”的提示词，本轮不重复修改。

## 验收

- 混合 `ASSESSMENT_CONTRACT_ERROR + TOO_LONG` 首次直接 `REWORK`，不消耗 Verifier retry。
- 单独 `ASSESSMENT_CONTRACT_ERROR` 仍先 `RETRY_VERIFIER`。
- 用户接受带风险草稿时，日志明确出现 task、choice 和最终 action。
- 现有 recovery、verifier 与 graph 测试保持通过。
