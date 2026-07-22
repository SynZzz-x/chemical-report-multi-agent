[AutoGLM](https://autoglm.zhipuai.cn/s/3b28ca0e-531b-4333-8000-e31f2d60fa28)

## 一、概览
![](https://cdn.nlark.com/yuque/__mermaid_v3/e5ad952664095ab8b8f331980da811a6.svg)

+ 背景：本项目采用“Intake → Planner → Worker → Verifier → Summarizer → Exit”的协作链条。每个环节产出明确中间结果，质量经校验后进入下一环节，最终合并为可交付文档并归档。

## 二、项目目标与交付物
+ 项目目标：将用户的自然语言需求拆解为若干可执行小任务，逐一完成并校验，集成为一份可读性强、可溯源、可复现的深度研究报告。
+ **主要交付物**：
    - 报告**文档**（word）：含摘要、研究问题、方法、数据分析、图表、发现与结论、参考来源、下一步建议。
    - 附件与素材：CSV 数据分析脚本/记录、生成的图表图片、抓取的网页要点摘要与来源列表。

## 三、工作流程与角色
+ 协作节奏：
    1. Intake：解析用户需求，澄清关键参数与约束，产出清晰的问题陈述，初始化 messages。
    2. Planner：将目标拆解为编号任务队列 tasks，并设定 cursor 初始位置。
    3. Worker：执行当前任务，产出 current_result（含文件路径、数据摘要、图表等）。
    4. Verifier：依据任务制定的判定流程进行质量校验；给出 decision（RETRY_WORKER/REPLAN/NEXT/DONE）与可操作 feedback；在 NEXT/DONE 时将 current_result 追加至 results 并推进 cursor。
    5. Summarizer：汇总 results 与 messages，形成最终报告与下一步建议。
    6. Exit：整理归档与恢复点（含版本与目录结构），便于后续检索与重启。

## 四、核心状态与数据规范
+ 必备状态字段及规范：
    - messages（List[Message]）：会话上下文。追加写入 Intake 问答与关键信息；Worker 可以追加工具轨迹；Summarizer 追加最终总结。使用“追加”聚合策略。
    - tasks（List[str]）：任务队列，由 Planner 生成；每项任务需有清晰描述、输入来源与输出期望。
    - cursor（int）：当前任务指针；Verifier 在 NEXT/DONE 推进；REPLAN 可重置为 0。
    - current_result（str）：Worker 对当前任务的产出，可包含文件路径、摘要、统计结果。
    - results（List[str]）：经验证累积的成果，由 Verifier 在 NEXT/DONE 追加 current_result。
    - decision（RETRY_WORKER/REPLAN/NEXT/DONE）：Verifier 的路由决策。
    - feedback（str）：Verifier 面向返工或重计划的可操作建议。
+ 数据约束与兼容性：
    - 上传数据目前仅支持 CSV。
    - 所有外部信息需保留来源（URL、访问日期、标题/站点名），便于审计与复现。
    - 图表以 PNG/SVG 输出，并在报告中配备清晰标题、坐标轴标注与简短说明。

## 五、任务分析
A. Intake

+ 任务 A1：需求澄清与问题陈述
    - 行动：与用户进行 3–5 个针对性问答，明确研究主题、数据范围、关键指标、时限、输出格式与风格。
    - 产出：问题陈述 v1（写入 messages）；明确输入资源清单。

B. Planner

+ 任务 B1：任务队列设计
    - 行动：将目标拆解成 6–10 个原子任务
    - 产出：tasks 队列与初始 cursor；每项任务描述需具体可执行。

C. Worker

+ 任务 C1：CSV 读取
    - 行动：读取 CSV，输出列名、类型、样本行。
    - 产出：结构报告 current_result；
+ 任务 C2：图表生成
    - 行动：根据 Planner 指定的分析目标，生成折线图/柱状图/散点图等；保存图片。
    - 产出：图表文件（PNG/SVG）与生成说明（参数、数据切片、图例）。
+ 任务 C3：搜索与抓取
    - 行动：用 Bing/Brave 搜索，使用 firecrawl 抓取重点网页内容；写要点摘要与来源。
    - 产出：摘要段落、引用列表（含 URL、标题/站点、访问日期）、去重后的要点合集。

D. Verifier

+ 任务 D1：质量校验与路由决策
    - 行动：对每个 current_result 验收；输出 decision 与反馈。
    - 产出：decision、feedback；在 NEXT/DONE 时将 current_result 加入 results 并推进 cursor。
+ 任务 D2：重计划触发与建议
    - 行动：当信息缺失或依赖错误时，提出 REPLAN 建议。
    - 产出：重计划建议（反馈中列明问题点与改动）。

E. Summarizer

+ 任务 E1：报告整编与写作
    - 行动：汇总 results 与 messages，形成报告初稿；统一术语、风格；插入图表与引用。
    - 产出：报告 v1（结构：摘要、问题、方法、数据与分析、发现、图表、参考、下一步建议）。

F. Exit / Memory

+ 任务 F1：归档与恢复点(checkPoint)



## 六、分工
### 分工详情
组 1：需求接入与总结（Intake／路由／Memory／Summarizer）

+ 职责
    - 与用户澄清需求，形成问题陈述（写入 messages），维护项目记忆与路由。
    - 汇总各阶段成果，编写报告 v1，完成归档与可恢复点。

组 2：规划与验证（Planner／Verifier）

+ 职责
    - 拆解目标为可执行 tasks（含输入来源与输出期望），设定初始 cursor。
    - 质量校验 current_result，给出 decision（RETRY_WORKER／REPLAN／NEXT／DONE）与可操作反馈；在 NEXT/DONE 时推进 cursor 并累计 results。

组 3：执行与工具（Worker＋工具操作）

+ 职责
    - 按 tasks 执行：CSV 读取与结构确认、可能需要收集用户信息、图表生成（PNG/SVG）、搜索与抓取。
    - 产出 current_result（含文件路径、摘要、统计/图表与生成说明），并在 messages 中记录工具轨迹。

### 名单
+ 10-19分组（项目初始）

| 分组 | 组长 |  |  | |
| :---: | --- | --- | --- | --- |
| 组1 | 黄佳俊 | 滕运韬 | 林硕 | 陈思彤 |
| 组2 | 张斯盈、韩卓龙 | 席书迪 | 谷唯一 | |
| 组3 | 郑沛豪 | 高明钰 | 段佳俊 | 杨吉骁 |


## 七、参考链接
[LangGraph](https://langchain-ai.github.io/langgraph/)

[Components | 🦜️🔗 LangChain](https://python.langchain.com/docs/integrations/components/)

