# 有界计划恢复与局部修补设计

## 1. 背景与问题

当前自动审核流程把 `BLOCKED` 及若干宽泛问题码直接转换为 `REPLAN`。随后 Planner 重新生成整套任务，将 `cursor` 重置为 0。该行为无法解决知识库证据不足或外部文件缺失，还会重复执行已通过章节，并可能让新旧 `task_id` 与结果发生冲突。

本次修改的目标是明确区分内容返工、证据恢复、局部计划修补和用户阻塞，移除自动执行过程中的全量重规划。

## 2. 核心原则

1. Verifier 只报告审核事实，不直接控制工作流。
2. 工作流动作由确定性的 DecisionPolicy 根据结构化问题类型、计数器和运行状态选择。
3. 执行期只允许局部 `PLAN_PATCH`，禁止自动重写整套任务或将游标重置为 0。
4. 已通过且不受补丁影响的结果必须保留。
5. 无法通过现有资源或自动工具解决的阻塞必须进入 `NEEDS_USER_INPUT`，不得通过重复调用模型掩盖。
6. `FULL_REPLAN` 只用于用户主动改变整体目标，并重新进入人工计划确认。

## 3. 节点职责

### 3.1 Verifier

Verifier 输出结构化审核结果：

- `status`: `PASS`、`FAILED` 或 `BLOCKED`；
- `issues`: 每项包含稳定的问题分类、描述、建议、依据和严重程度；
- `requirements_met`: 已满足要求；
- `requirements_missing`: 未满足要求。

Verifier 不再输出或决定 `REPLAN`。即使模型返回路由建议，系统也不直接采用。

### 3.2 DecisionPolicy

DecisionPolicy 是不调用 LLM 的确定性节点。它将审核结果映射为以下动作：

- `PASS`: 提交当前结果并继续；
- `REWORK`: 携带审核意见返工当前任务；
- `EVIDENCE_RECOVERY`: 对当前任务执行一次证据恢复；
- `PLAN_PATCH`: 请求 PlanPatcher 对明确受影响的任务做局部修改；
- `NEEDS_USER_INPUT`: 暂停工作流并展示具体阻塞；
- `ACCEPT_WITH_WARNING`: 达到内容返工上限后保留结果、记录警告并继续；
- `DONE`: 提交最后一个任务并进入 Summarizer。

### 3.3 EvidenceRecovery

EvidenceRecovery 不修改任务结构。它为当前任务生成结构化恢复指令，要求 Worker：

1. 根据缺失概念改写检索词；
2. 扩大 RAG 检索覆盖；
3. 在任务允许公开网络补充时调用安全网络来源；
4. 保留来源和证据覆盖信息；
5. 不得用无来源内容填补证据缺口。

每个任务最多自动恢复一次。恢复后仍无法满足强制证据要求时进入 `NEEDS_USER_INPUT`。

### 3.4 PlanPatcher

PlanPatcher 只生成结构化补丁，不生成完整任务列表。允许的操作为：

- 更新当前或后续任务的描述、检索参数、生成要求和已有资源分配；
- 调整未完成任务的顺序；
- 在当前任务之前插入必要的前置任务；
- 明确使受影响的已提交结果失效。

禁止的操作为：

- 自动删除已完成任务；
- 修改未列入 `affected_task_ids` 的任务；
- 引用系统中不存在的资源；
- 将执行位置无条件重置为第一个任务；
- 以新任务列表替换当前完整计划。

每个任务最多应用一次计划补丁，每个 Job 最多应用三次。超过限制或补丁校验失败时进入 `NEEDS_USER_INPUT`。

### 3.5 用户触发的 Full Replan

`FULL_REPLAN` 不属于自动审核路由。只有用户明确修改报告目标、范围或章节结构时才能触发。该操作生成新的完整计划修订，并必须经过 Planner Confirm 后才能执行。

## 4. 问题分类与路由规则

| 问题分类 | 示例 | 动作 |
| --- | --- | --- |
| `CONTENT_DEFECT` | 字数不足、遗漏要点、格式错误、图表未生成 | `REWORK` |
| `EVIDENCE_GAP` | RAG 命中不足、关键结论缺少来源、检索覆盖不足 | `EVIDENCE_RECOVERY` |
| `LOCAL_PLAN_DEFECT` | 已有文件未分配、依赖顺序错误、当前任务粒度不可执行 | `PLAN_PATCH` |
| `EXTERNAL_BLOCKER` | 必需文件不存在、权限缺失、需求矛盾需用户选择 | `NEEDS_USER_INPUT` |

`MISSING_RESOURCE` 不再直接等于计划错误：资源存在但未分配属于 `LOCAL_PLAN_DEFECT`；资源根本不存在属于 `EXTERNAL_BLOCKER`；知识证据检索不足属于 `EVIDENCE_GAP`。

## 5. 状态契约

在保留当前串行 `cursor` 工作流的前提下，新增或规范以下状态：

- `workflow_action`: 当前确定性动作；
- `plan_revision`: 当前计划修订号，初始为 1；
- `task_revisions`: 各任务自身修订号；
- `task_retry_count`: 各任务内容返工次数；
- `evidence_recovery_count`: 各任务证据恢复次数；
- `task_patch_count`: 各任务局部补丁次数；
- `job_patch_count`: Job 已应用补丁总数；
- `pending_user_action`: 暂停原因、所需输入、受影响任务及可选处理方式；
- `plan_patch_history`: 补丁版本、原因、操作、影响任务和恢复位置；
- `verification_warnings`: 达到上限后被保留的审核警告。

任务 ID 在 Job 生命周期内保持稳定。插入任务使用不会与现有任务重复的新 ID。结果记录任务 ID、任务修订号和计划修订号，避免旧结果被错误复用。

## 6. Plan Patch 格式与校验

补丁包含：

- `base_plan_revision`；
- `reason_code` 和 `reason`；
- `affected_task_ids`；
- `operations`；
- `resume_task_id`；
- `expected_resolution`。

应用前必须确定性校验：

1. `base_plan_revision` 与当前状态一致；
2. 所有被引用任务 ID 存在，插入任务 ID 唯一；
3. 所有资源来自当前 Job 的资源清单；
4. 操作范围不超出 `affected_task_ids`；
5. 不修改未声明的已完成任务；
6. `resume_task_id` 存在且是最早受影响任务；
7. 补丁明确说明预计如何解除原阻塞。

校验通过后原子应用补丁，将 `plan_revision` 加一，只清理受影响结果，并从 `resume_task_id` 恢复。校验失败不做部分写入。

## 7. 有界恢复策略

- 内容问题：每个任务最多自动 `REWORK` 两次；之后 `ACCEPT_WITH_WARNING`。
- 证据问题：每个任务最多一次 `EVIDENCE_RECOVERY`；仍无法满足强制要求则 `NEEDS_USER_INPUT`。
- 计划问题：每个任务最多一次 `PLAN_PATCH`，每个 Job 最多三次；超限则 `NEEDS_USER_INPUT`。
- 外部阻塞：直接 `NEEDS_USER_INPUT`，不消耗内容重试次数。
- 任何路径都不得回退为自动全量重规划。

## 8. 图路由调整

自动模式调整为：

```text
Worker -> Verifier -> DecisionPolicy
  PASS/NEXT -----------------------> Planner(推进游标)
  REWORK --------------------------> Worker
  EVIDENCE_RECOVERY -> Recovery ---> Worker
  PLAN_PATCH -> PlanPatcher -> PatchValidator -> Worker
  NEEDS_USER_INPUT ----------------> 暂停
  DONE ----------------------------> Summarizer
```

删除自动图中的 `Verifier -> REPLAN -> Planner` 路由。手动模式保留用户要求整体重规划的入口，但映射为 `FULL_REPLAN` 并进入 Planner Confirm。

## 9. 错误处理与兼容性

- 旧 checkpoint 缺少新增字段时使用安全默认值，不要求删除 SQLite 数据库。
- 旧状态中的 `replan_count` 只作兼容读取，不再参与自动路由。
- Verifier JSON 解析失败按审核服务错误处理：有限重试或暂停，不得判定计划错误。
- PlanPatcher 模型输出非法 JSON、非法操作或幻觉资源时，PatchValidator 拒绝补丁并进入 `NEEDS_USER_INPUT`。
- 应在控制台和 JSONL 日志中记录任务、问题分类、策略动作、各类计数、计划修订号和补丁结果。

## 10. 验收与测试

需要覆盖以下回归场景：

1. `BLOCKED + EVIDENCE_GAP` 进入证据恢复，不进入 Planner；
2. 资源存在但未分配时只修改对应任务；
3. 资源不存在时暂停请求用户输入；
4. 已通过 T1、T2 后修补 T3，不重新执行 T1、T2；
5. 补丁使用旧计划版本、未知任务或未知资源时被拒绝；
6. 内容返工、证据恢复和计划补丁各自遵守独立上限；
7. 旧 checkpoint 可以恢复并使用默认状态；
8. 自动工作流不存在 `REPLAN -> Planner` 路由；
9. 用户主动 `FULL_REPLAN` 时仍进入计划确认；
10. 本次服务器日志中的 T2 场景不会将 cursor 重置为 0。

不依赖真实 DeepSeek、RAG 或网络 API 完成自动化测试，使用确定性桩验证状态和路由。

## 11. 非目标

本次不把串行 `cursor` 工作流整体改造成 DAG，不实现任务并行调度，也不重构与计划恢复无关的 RAG、图表和报告生成逻辑。这些工作可在局部恢复稳定后单独推进。
