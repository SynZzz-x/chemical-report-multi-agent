# 报告流水线稳定化与质量审查重构设计

## 1. 背景

项目已经具备 DeepSeek 统一配置、SQLite Checkpoint/Store、混合 RAG、证据引用、概念关系图、Streamlit 界面和有界恢复机制，继续推倒重建的成本高于收益。本阶段选择在现有 `Planner -> Worker -> Verifier -> Summarizer` 基础上演进，先把报告流水线修稳定，再为后续拆分专业 Agent 和升级异常调查工作台保留接口。

服务器实测暴露了四类核心问题：

1. 九个任务中仅 T1 执行，随后提前进入 Summarizer，说明执行进度仍依赖易漂移的 `cursor`、消息或 Planner 副作用；
2. Worker 返工时重复同一任务，但用户无法区分正常有限返工和系统重复执行；
3. Summarizer 生成普通 Markdown 时错误启用 JSON 模式，DeepSeek 返回 HTTP 400；
4. PDF 表格解析没有识别转义竖线，导致列数膨胀和 ReportLab 布局失败，失败后的 PDF 仍可能显示为附件。

本设计称为“方案 B+”：保留报告生成主线，重构任务推进、质量审查、局部恢复和报告交付，不在本阶段建设完整的化工异常案例管理平台。

## 2. 产品定位和首期场景

首期服务于聚乙烯产品质量异常分析报告，覆盖 MFR/MI、密度、鱼眼/凝胶、灰分/挥发分、外观和粒径分布等问题。

输入范围为：

- 企业 PDF、DOCX、SOP、操作手册和历史报告；
- 用户手动上传的 CSV/Excel 过程或质量数据；
- 经用户对当前任务明确授权后检索的公开网络资料。

公开网络资料只允许来自政府、标准组织、设备厂商、学术机构等可信来源，必须与企业内部证据分开标记。公开资料不能替代企业 SOP，也不能单独支撑装置级具体操作指令。

系统可以输出原因优先级、证据、后续核查顺序和方向性建议。没有企业 SOP、批准范围或人工确认时，不输出具体调参幅度，不直接连接或控制 DCS、MES、LIMS 等生产系统。

## 3. 目标与非目标

### 3.1 目标

1. 多任务报告严格按任务台账推进，不跳过任务、不重复提交同一执行版本；
2. Worker、QualityReview 和流程路由各自职责单一；
3. 内容返工、证据恢复、技术故障和用户阻塞使用不同恢复路径；
4. 完整重规划只能由用户主动触发和确认；
5. 报告只汇总已经通过审查的产物，并准确反映 DOCX、PDF 和附件的生成状态；
6. 任务、产物、审查和报告版本可以通过 SQLite 恢复和审计；
7. Worker 后续可以按任务类型拆成多个专业 Agent，而不改变顶层流程契约。

### 3.2 非目标

- 不建设完整的异常 Case 管理、班组协同、告警处置或工单系统；
- 不直接接入或控制现场生产系统；
- 不把串行任务改造成并行 DAG 调度；
- 不照搬 DATAGEN 的动态 Supervisor；
- 不在本阶段一次性拆分全部 Worker 工具；
- 不允许自动流程对整个任务计划进行全量重规划。

## 4. 总体架构

自动模式调整为：

```text
Intake
  -> Planner
  -> PlannerConfirm
  -> TaskController
  -> Worker
  -> QualityReview
  -> DecisionPolicy
       PASS              -> TaskController
       REVISE            -> Worker
       EVIDENCE_REQUIRED -> EvidenceRecovery -> Worker
       LOCAL_PLAN_PATCH  -> PlanPatcher -> TaskController
       BLOCKED           -> HumanReview
       REVIEW_RETRY       -> QualityReview
  -> Summarizer
  -> HumanReview
  -> Exit
```

人工计划审核模式复用相同的 TaskController、Artifact 和 ReportManifest 契约，只把自动 QualityReview 的最终判断替换或补充为人工判断，避免维护两套任务推进语义。

## 5. 组件职责

### 5.1 Planner

Planner 只负责根据用户目标生成初始 `ExecutionPlan`，或在用户主动要求并确认后生成完整的新计划版本。Planner 不推进任务游标、不判断 Worker 质量、不处理模型重试，也不能由自动 Verifier 触发。

### 5.2 TaskController

TaskController 是不调用 LLM 的确定性节点，负责：

- 根据任务台账选择序号最小、依赖已满足且尚未通过的任务；
- 将任务从 `PENDING` 转为 `RUNNING`；
- 在全部任务 `PASSED` 后才允许进入 Summarizer；
- 从 SQLite 恢复后重新计算当前任务；
- 拒绝使用旧 Artifact 或旧 ReviewRecord 推进任务。

`cursor` 仅作为兼容字段和界面投影，不再作为进度事实来源。任务台账是唯一进度真相。

### 5.3 Worker

Worker 一次只执行一个 TaskRecord，产出一个新的 Artifact。Worker 不判断自己的结果是否合格，不更新下一任务，不触发完整重规划。

本阶段保留 Worker 顶层节点，但在内部定义统一处理器接口，使任务可以逐步交给 `EvidenceHandler`、`DataAnalysisHandler`、`VisualizationHandler` 和 `SectionWritingHandler`。未来这些处理器可以升级成独立 Agent，而无需改变 TaskController、QualityReview 或 DecisionPolicy 的数据契约。

### 5.4 QualityReview

QualityReview 借鉴 DATAGEN 将质量审查作为独立 Agent 的思路，但不照搬其二值输出和消息位置路由。

审查分两层：

1. 确定性校验器检查结构完整性、必需表格/图形、引用存在性、证据引用可解析、数据与图表一致性、文件状态和安全边界；
2. LLM 审查切题程度、因果逻辑、矛盾、原因排序、证据支持强度和建议可执行性。

QualityReview 只生成 ReviewRecord。它不能修改 Artifact、调用 Worker 工具、决定路由或触发 replan。

### 5.5 DecisionPolicy

DecisionPolicy 不调用 LLM，根据 ReviewRecord 的结构化问题类型、严重程度和次数上限选择确定性动作。问题绑定显式 `responsible_handler`，不得通过 `messages[-2]` 等上下文位置猜测返工对象。

### 5.6 Summarizer

Summarizer 只读取各任务最新且通过审查的 Artifact。它负责确定性组装章节、引用清单和证据追溯表，再按需调用非 JSON 模式的 LLM 生成自然语言报告评价。

DOCX 和 PDF 独立生成、独立记录状态、独立重试。某一格式失败不得重新执行业务任务，也不得将不存在或失败的文件显示为可下载附件。

### 5.7 HumanReview

HumanReview 处理安全边界、重试超限、外部资料缺失、用户修改审查意见、完整重规划确认和最终报告确认。人工接受不删除原审查问题，而是记录接受人、理由和时间。

## 6. 状态与持久化契约

### 6.1 ExecutionPlan

```text
plan_id
plan_revision
goal
tasks[]
created_at
created_by
```

局部返工不增加 `plan_revision`。只有用户确认完整重规划后才创建新版本。

### 6.2 TaskRecord

```text
task_id
sequence
title
description
requirements
required_artifacts
allowed_sources
status
attempt_count
active_artifact_id
dependencies
```

任务状态只允许：

```text
PENDING
RUNNING
REVISE_REQUIRED
EVIDENCE_REQUIRED
PASSED
BLOCKED
```

### 6.3 Artifact

```text
artifact_id
task_id
attempt_no
artifact_type
producer
content
tables
figures
citations
evidence_refs
source_scope
created_at
supersedes
```

每次返工创建新 Artifact，通过 `supersedes` 关联旧版本。旧版本保留用于审计，不覆盖写入。

### 6.4 ReviewRecord

```text
review_id
task_id
artifact_id
status
issues[]
quality_dimensions
reviewer
created_at
```

`status` 允许 `PASS`、`REVISE`、`BLOCKED`、`HUMAN_REVIEW`。每个 issue 至少包含：

```text
code
category
severity
description
evidence_refs
responsible_handler
revision_instruction
```

问题分类包括：

- `CONTENT_DEFECT`；
- `EVIDENCE_GAP`；
- `DATA_DEFECT`；
- `VISUAL_DEFECT`；
- `WORKER_FAILURE`；
- `LOCAL_PLAN_DEFECT`；
- `SAFETY_BOUNDARY`；
- `REQUIREMENT_MISSING`；
- `EXTERNAL_BLOCKER`；
- `REVIEW_FAILURE`。

### 6.5 ReportManifest

```text
report_id
included_artifact_ids
docx_status
docx_path
pdf_status
pdf_path
generation_errors
created_at
```

大型 DOCX、PDF 和图片继续保存为文件。LangGraph Store 保存计划、任务、产物、审查和报告清单的结构化记录；Checkpoint 保存图执行位置和运行状态。数据库只存文件路径、校验值和元数据，不存大型二进制内容。

## 7. 任务推进规则

1. TaskController 选择第一个依赖满足且状态不为 `PASSED` 的任务；
2. Worker 使用 `job_id + task_id + attempt_no` 作为幂等执行标识；
3. Worker 成功后写入 Artifact，并把该 ID 交给 QualityReview；
4. ReviewRecord 必须绑定当前 TaskRecord 的 `active_artifact_id`；
5. `PASS` 将任务设为 `PASSED`，TaskController 再选择下一任务；
6. `REVISE` 或 `EVIDENCE_REQUIRED` 只影响当前任务；
7. 只要存在非 `PASSED` 任务，Summarizer 就不能启动；
8. 新 Artifact 产生后，旧 ReviewRecord 不再参与路由，但继续保留；
9. 页面刷新、节点重放或服务重启发现相同幂等键已经提交时，复用已提交 Artifact，不重复执行；
10. 任务推进和 Artifact 激活必须作为同一逻辑提交完成，避免结果已保存但状态未推进的半完成状态。

## 8. 审查与恢复策略

| 问题 | 动作 | 自动上限 |
| --- | --- | ---: |
| Worker 临时调用失败 | 重试当前 Worker | 1 |
| `CONTENT_DEFECT` | 按明确意见局部返工 | 2 |
| `EVIDENCE_GAP` | 补充 RAG 或已授权网络证据后局部返工 | 1 |
| `DATA_DEFECT` | 返回数据分析处理器 | 1 |
| `VISUAL_DEFECT` | 只重新生成受影响图表 | 1 |
| `REQUIREMENT_MISSING` | 补齐当前任务缺失项 | 2 |
| `SAFETY_BOUNDARY` | 进入人工审核 | 0 |
| `EXTERNAL_BLOCKER` | 等待资料、权限或用户选择 | 0 |
| `REVIEW_FAILURE` | 只重试 QualityReview | 1 |
| DOCX/PDF 生成失败 | 单独重试对应格式 | 1 |

达到上限后进入 `BLOCKED -> HumanReview`，不得强制通过。该规则替代《有界计划恢复与局部修补设计》中内容返工超限后 `ACCEPT_WITH_WARNING` 的行为；警告可以保留，但不能自动把任务标记为通过。

QualityReview 服务故障必须分类为 `REVIEW_FAILURE`，不得把 Worker 产物误判为内容失败。Summarizer 或某种报告格式失败不得重新执行 Planner、Worker 或 QualityReview。

`LOCAL_PLAN_DEFECT` 仅允许调用现有 PlanPatcher 修补当前或未完成任务的资源分配、依赖顺序和局部要求。补丁必须继续遵守既有校验与次数上限，完成后返回 TaskController；它不是完整 replan，不能替换整套计划或重置已经通过的任务。

## 9. Replan 的唯一语义

完整 replan 仅表示：用户目标、任务拆分或任务依赖关系已经不再成立，需要生成新的完整 ExecutionPlan。

以下情况均不属于 replan：

- 字数、格式或章节内容不合格；
- 引用或知识证据不足；
- 表格或图形缺失；
- LLM、Embedding、网络或文件生成临时失败；
- 某个任务需要调整检索词或补充局部要求。

自动流程不得触发完整 replan。用户在界面查看影响范围并二次确认后，Planner 才能生成新计划版本。旧计划、已通过 Artifact 和审查历史继续保留。

## 10. Streamlit 工作区

方案 B+ 仍以报告任务为主要对象。页面提供：

1. 左侧任务栏：历史任务、恢复入口、新建任务、模型/RAG 状态、审核模式和网络授权；
2. 顶部进度：总任务数、已通过数量、当前任务、状态和计划版本；
3. 当前任务工作区：任务要求、最新产物、内部证据、公开证据、图表、QualityReview 结果和历史返工版本；
4. 人工操作：接受、局部返工、补充资料、修改审查意见、阻塞任务、请求完整重规划；
5. 报告交付：分别展示 DOCX、PDF、引用清单和证据追溯表的真实生成状态。

界面必须区分“重试模型调用”“返工当前任务”“重新生成图表”“重新生成报告”和“完整重规划”，并显示每次执行的 attempt、Artifact 和审查记录，让用户能区分有限返工与异常重复。

## 11. PDF 和报告生成约束

1. 普通 Markdown 或报告评价调用 LLM 时必须使用 `json_mode=False`；只有明确要求 JSON 对象的调用才能开启 JSON 模式；
2. Markdown 表格解析必须识别 `\|` 等转义字符，不能使用未处理的 `split('|')`；
3. 超宽表格应采用可预测的降级策略，例如横向页面、列宽上限、字号下限或拆表，不得让 ReportLab 抛出负可用宽度；
4. ReportManifest 只登记实际存在且校验成功的文件；
5. PDF 失败时可以交付成功的 DOCX，但必须展示 PDF 错误和单独重试入口；
6. Summarizer 不得为了补齐失败章节自行生成未经 QualityReview 的新章节内容。

## 12. 测试与验收

### 12.1 自动化测试层次

单元测试覆盖：

- TaskController 的任务选择、依赖判断和完成判断；
- Artifact 与 ReviewRecord 的版本绑定；
- DecisionPolicy 的问题分类、责任处理器和次数上限；
- ReportManifest 的文件状态；
- Markdown 表格中的转义竖线和超宽表格处理；
- LLM JSON 模式的调用边界。

组件测试覆盖：

- Worker 一次只接收并提交一个任务；
- 新 Artifact 使旧 ReviewRecord 退出有效路由；
- QualityReview 故障只重试审查节点；
- Summarizer 只接收全部 `PASSED` 的任务产物；
- 人工模式和自动模式使用同一任务台账语义。

端到端回归测试使用确定性 LLM/RAG 桩，至少覆盖：

1. 九个任务按顺序全部完成，每个任务恰好提交一个通过版本；
2. T3 内容失败两次只返工 T3，T1、T2 不重复；
3. T3 补充证据后继续 T4，不触发完整 replan；
4. 任一任务未通过时 Summarizer 不启动；
5. 服务在 T3 Worker、QualityReview 和 Summarizer 三个位置重启后均能从 SQLite 正确恢复；
6. DeepSeek 普通报告生成不携带 JSON response format；
7. 包含转义竖线的证据表不会被解析成异常列数；
8. PDF 失败时不产生虚假附件，DOCX 仍可下载；
9. 重试超限进入 HumanReview，不强制通过；
10. 自动图中不存在通往完整 replan 的边。

### 12.2 首期业务验收

使用预设的聚乙烯质量异常案例进行人工验收。系统必须：

- 给出前三位候选原因；
- 每个主要结论都能追溯到企业资料、上传数据或明确标记的公开来源；
- 给出按优先级排列的下一步核查项；
- 只给出方向性建议，不越过企业 SOP 和批准范围；
- 允许工程师修改、接受或驳回审查意见；
- 完整保存计划、任务版本、证据、审查、人工操作和最终报告；
- 不出现任务跳过、无限循环、未经确认的完整重规划或失败附件伪成功。

在具备企业历史案例后，再增加“已知根因进入前三位的比例”和“工程师诊断耗时变化”等业务指标；这些指标不作为本次代码稳定化的阻塞条件。

## 13. 兼容与迁移

- 保留当前 `tasks`、`cursor`、`results` 等字段作为迁移期读写适配层；
- 旧 checkpoint 缺少任务台账、Artifact 或 ReviewRecord 时，从现有任务和结果确定性构造默认记录；
- 不要求用户删除现有 SQLite 数据库；
- 新流程不再把 `replan_count`、消息发送者或聊天文本作为任务推进依据；
- 旧 `Verifier` 输出通过适配器转换为 ReviewRecord，完成迁移后再移除旧契约；
- 现有 DeepSeek、RAG、证据、概念图和文件生成模块通过明确接口复用，不进行无关重写。

## 14. 后续演进

当 Worker 的单体职责成为主要维护瓶颈时，将内部处理器依次升级为 EvidenceAgent、DataAnalysisAgent、VisualizationAgent 和 SectionWritingAgent。各 Agent 仍输出 Artifact，并接受相同 ReviewRecord 和 DecisionPolicy 约束。

当报告任务流水线稳定且企业需要持续管理异常调查时，再在本方案的 ExecutionPlan、Artifact、ReviewRecord 和 HumanReview 之上增加 Case、时间线、假设状态和协同功能，演进到方案 A，而不是再次重写底层持久化与质量控制。
