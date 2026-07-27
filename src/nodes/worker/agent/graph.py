from typing import TypedDict, List, Dict, Any, Optional, Callable, Type
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import StateGraph, END
from ....config import get_app_config
from ....llm import get_llm
from ....utils.path_manager import get_session_cache_dir
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import json
import os
import sys
import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import importlib
import inspect
import hashlib
import shutil
import re
import ast

# 添加工具目录到路径
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

# 针对 Windows 平台的异步子进程支持修复
if sys.platform == 'win32':
    try:
        # 设置 ProactorEventLoopPolicy 以支持 subprocess
        if isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass


# ==================== 0. 全局路径配置 ====================
# 获取当前文件所在目录: src/nodes/worker/agent
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录: src/nodes/worker/agent -> worker -> nodes -> src -> Agent (Root)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))


# ==================== 1. 配置类 ====================
@dataclass
class WorkerConfig:
    """Worker配置类"""
    API_KEY: str = field(
        default_factory=lambda: get_app_config().deepseek_api_key or ""
    )
    BASE_URL: str = field(
        default_factory=lambda: get_app_config().deepseek_base_url
    )
    LLM_MODEL: str = field(
        default_factory=lambda: get_app_config().deepseek_model
    )
    DASHSCOPE_API_KEY: str = field(
        default_factory=lambda: get_app_config().dashscope_api_key or ""
    )
    DASHSCOPE_EMBEDDING_MODEL: str = field(
        default_factory=lambda: get_app_config().dashscope_embedding_model
    )
    TEMPERATURE: float = 0.7
    TIMEOUT: int = 60
    MAX_RETRIES: int = 3

    # 路径配置 (统一使用绝对路径)
    BASE_DIR: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "cache"))
    CHARTS_DIR: str = ""
    REPORT_DIR: str = ""
    OUTPUT_DIR: str = ""
    SPIDER_DIR: str = ""
    LOGS_DIR: str = ""

    # 工具目录和知识库目录
    TOOLS_DIR: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "src", "nodes", "worker", "tools"))
    KNOWLEDGE_BASE_DIR: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "data", "chemical_kb"))

    # 工具配置
    ENABLED_TOOLS: List[str] = field(default_factory=lambda: [
        "ChemicalKnowledgeBaseTool",
        "CSVTool",
        "ChartTool",
        "SpiderTool"
    ])

    # 系统设置
    MAX_TOOL_ITERATIONS: int = 15
    TOOL_CALL_TIMEOUT: int = 30
    MAX_CONCURRENT_TOOLS: int = 3

    # 图表生成控制
    MAX_CHARTS_PER_TASK: int = 6
    MAX_CHARTS_PER_DATASET: int = 2
    ENABLE_CHART_GENERATION: bool = True
    CHART_CACHE_ENABLED: bool = True
    CHART_CACHE_EXPIRY_HOURS: int = 2

    # 报告生成控制
    GENERATE_SINGLE_REPORT: bool = False
    CLEAN_REPORT_ON_START: bool = False

    # 知识库配置
    KNOWLEDGE_BASE_ENABLED: bool = True
    AUTO_LOAD_RESOURCES: bool = True

    # 爬虫配置
    SPIDER_ENABLED: bool = True
    MAX_SPIDER_RESULTS: int = 3
    SPIDER_TIMEOUT: int = 30

    def __post_init__(self):
        """确保目录存在并初始化路径"""
        # 如果子路径未设置，则根据 BASE_DIR 初始化
        if not self.CHARTS_DIR:
            self.CHARTS_DIR = os.path.join(self.BASE_DIR, "charts")
        if not self.REPORT_DIR:
            self.REPORT_DIR = os.path.join(self.BASE_DIR, "report")
        if not self.OUTPUT_DIR:
            self.OUTPUT_DIR = os.path.join(self.BASE_DIR, "output")
        if not self.SPIDER_DIR:
            self.SPIDER_DIR = os.path.join(self.BASE_DIR, "worker_scrape_results")
        if not self.LOGS_DIR:
            self.LOGS_DIR = os.path.join(PROJECT_ROOT, "logs", "worker")

        print(f"📁 配置信息:")
        print(f"   BASE_DIR: {self.BASE_DIR}")
        print(f"   CHARTS_DIR: {self.CHARTS_DIR}")
        print(f"   REPORT_DIR: {self.REPORT_DIR}")

        for dir_path in [self.CHARTS_DIR, self.REPORT_DIR, self.OUTPUT_DIR, self.SPIDER_DIR, self.LOGS_DIR, self.KNOWLEDGE_BASE_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ 确保目录存在: {dir_path}")

        # 清理旧报告（如果配置了）
        if self.CLEAN_REPORT_ON_START and os.path.exists(self.REPORT_DIR):
            for file_name in os.listdir(self.REPORT_DIR):
                file_path = os.path.join(self.REPORT_DIR, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"🗑️ 清理报告文件: {file_name}")
                except Exception as e:
                    print(f"⚠️ 清理报告文件 {file_name} 失败: {e}")


# ==================== 2. 数据模型 ====================

class Task(TypedDict):
    """任务数据结构"""
    task_id: str
    task_name: str
    task_type: str
    task_description: str
    use_rag: bool
    generate_table: bool
    generate_figure: bool
    use_resources: List[str]
    query: Optional[str]
    tool_requirements: Optional[List[str]]
    priority: int = 1
    knowledge_base_loaded: bool = False


class TaskResult(TypedDict):
    """任务结果数据结构"""
    task_id: str
    section_name: str
    text_output: str
    status: str
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    sources_used: List[str]
    figures_generated: int
    word_count: int
    generated_at: str
    execution_time: float
    tool_calls: List[Dict[str, Any]]
    tool_usage_stats: Dict[str, int]
    error: Optional[str]
    knowledge_base_used: bool = False
    spider_results_used: bool = False


class State(TypedDict):
    """全局状态定义"""
    messages: List[AnyMessage]
    tasks: List[Task]
    current_task: Optional[Task]
    current_result: Optional[TaskResult]
    all_results: List[TaskResult]
    cursor: int
    worker_state: Dict[str, Any]
    tool_execution_history: List[Dict[str, Any]]
    available_tools: List[Dict[str, Any]]
    knowledge_base_initialized: bool = False
    spider_initialized: bool = False


# ==================== 3. 工具基类和接口 ====================

class ToolExecutionError(Exception):
    """工具执行异常"""
    pass


class BaseWorkerTool(ABC):
    """工具基类 - 所有自定义工具必须继承此类"""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.name = self.get_tool_name()
        self.description = self.get_tool_description()
        self.required_resources = self.get_required_resources()
        self.version = "1.0.0"

    @abstractmethod
    def get_tool_name(self) -> str:
        """返回工具名称"""
        pass

    @abstractmethod
    def get_tool_description(self) -> str:
        """返回工具描述"""
        pass

    @abstractmethod
    def get_args_schema(self) -> Type[BaseModel]:
        """返回参数模型"""
        pass

    @abstractmethod
    def execute(self, task: Task, **kwargs) -> Dict[str, Any]:
        """执行工具逻辑"""
        pass

    def get_required_resources(self) -> List[str]:
        """返回工具所需的资源类型"""
        return []

    def validate_task(self, task: Task) -> bool:
        """验证任务是否适合使用此工具"""
        if self.required_resources:
            task_resources = task.get("use_resources", [])
            for resource in task_resources:
                if isinstance(resource, str):
                    for req_type in self.required_resources:
                        if resource.endswith(f".{req_type}"):
                            return True
            return False
        return True

    def get_tool_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "required_resources": self.required_resources
        }

    def create_langchain_tool(self, task: Task) -> BaseTool:
        """创建LangChain兼容的工具"""

        def tool_function(**kwargs):
            """工具执行函数"""
            try:
                result = self.execute(task, **kwargs)
                if isinstance(result, dict):
                    if "figures" in result:
                        return json.dumps(result, ensure_ascii=False, indent=2)
                    elif "text_output" in result:
                        return result["text_output"]
                    elif "summary" in result:
                        return result["summary"]
                    elif "content" in result:
                        return result["content"]
                    else:
                        return json.dumps(result, ensure_ascii=False, indent=2)
                else:
                    return str(result)
            except Exception as e:
                return f"工具执行失败: {str(e)}"

        tool_function.__doc__ = self.description

        try:
            try:
                from langchain.tools import tool as tool_decorator
            except ImportError:
                from langchain_core.tools import tool as tool_decorator

            @tool_decorator(args_schema=self.get_args_schema())
            def decorated_tool(**kwargs):
                """工具执行函数"""
                return tool_function(**kwargs)

            decorated_tool.name = self.name
            return decorated_tool

        except Exception as e:
            print(f"⚠️ 使用@tool装饰器创建工具失败: {e}")

            class CustomTool(BaseTool):
                name: str = self.name
                description: str = self.description
                args_schema: Type[BaseModel] = self.get_args_schema()

                def _run(self, **kwargs):
                    """同步执行工具"""
                    return tool_function(**kwargs)

                async def _arun(self, **kwargs):
                    """异步执行工具"""
                    raise NotImplementedError("此工具不支持异步执行")

            return CustomTool()


# ==================== 4. 真实工具实现 ====================

class ChemicalKnowledgeBaseTool(BaseWorkerTool):
    """化工知识库工具 - 从知识库检索化工专业知识"""

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.knowledge_base = None
        self._initialized = False
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """初始化知识库"""
        try:
            from ..tools.ChemicalKnowledgeBase import ChemicalKnowledgeBase
            self.knowledge_base = ChemicalKnowledgeBase(
                dashscope_api_key=self.config.DASHSCOPE_API_KEY,
                embedding_model=self.config.DASHSCOPE_EMBEDDING_MODEL,
                persist_directory=self.config.KNOWLEDGE_BASE_DIR
            )
            self._initialized = True
            print(f"✅ {self.name} 初始化成功")
            print(f"   知识库路径: {self.config.KNOWLEDGE_BASE_DIR}")
        except ImportError as e:
            print(f"❌ 化工知识库工具导入失败: {e}")
            self._initialized = False
        except Exception as e:
            print(f"❌ 化工知识库初始化失败: {e}")
            self._initialized = False

    def get_tool_name(self) -> str:
        return "chemical_knowledge_base_tool"

    def get_tool_description(self) -> str:
        return """从化工专业知识库中检索相关技术资料、标准和规范。支持文档加载、知识检索和智能问答。"""

    def get_args_schema(self) -> Type[BaseModel]:
        class KnowledgeBaseArgs(BaseModel):
            operation: str = Field(
                default="query",
                description="操作类型：query（查询）/load（加载文档）/stats（统计信息）"
            )
            query: Optional[str] = Field(default=None, description="检索查询内容，当operation=query时需要")
            top_k: int = Field(default=5, ge=1, le=10, description="返回结果数量，范围1-10")
            similarity_threshold: float = Field(
                default=0.3,
                ge=0.1,
                le=1.0,
                description="相似度阈值，范围0.1-1.0"
            )
            generate_answer: bool = Field(default=True, description="是否生成智能答案")
            doc_type_filter: Optional[str] = Field(
                default=None,
                description="文档类型过滤器：patent/safety/process/equipment/material/standard/data/report"
            )

        return KnowledgeBaseArgs

    def validate_task(self, task: Task) -> bool:
        """验证任务是否适合使用此工具"""
        # 如果任务显式指定了使用RAG，则始终启用此工具（即使没有新上传的文档）
        if task.get("use_rag"):
            return True
        return super().validate_task(task)

    def get_required_resources(self) -> List[str]:
        return ["txt", "pdf", "docx", "md", "csv", "xlsx", "xls"]

    def execute(self, task: Task, **kwargs) -> Dict[str, Any]:
        if not self._initialized or not self.knowledge_base:
            return {
                "error": "化工知识库工具未初始化",
                "suggestion": "请检查ChemicalKnowledgeBase模块是否正确安装"
            }

        try:
            operation = kwargs.get("operation", "query")

            if operation == "load":
                return self._load_documents_to_kb(task, kwargs)
            elif operation == "stats":
                return self._get_knowledge_base_stats()
            else:
                return self._query_knowledge_base(task, kwargs)

        except Exception as e:
            raise ToolExecutionError(f"化工知识库操作失败: {str(e)}")

    def _load_documents_to_kb(self, task: Task, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """加载文档到知识库"""
        resources = task.get("use_resources", [])
        supported_extensions = ['.txt', '.pdf', '.docx', '.md', '.csv', '.xlsx', '.xls']
        document_files = [
            r for r in resources
            if isinstance(r, str) and os.path.splitext(r.lower())[1] in supported_extensions
        ]

        if not document_files:
            return {
                "error": "未找到可加载的文档文件",
                "suggestion": "请确保任务资源中包含支持的文档类型"
            }

        print(f"📚 正在加载 {len(document_files)} 个文档到知识库...")
        load_result = self.knowledge_base.load_documents(document_files)
        task["knowledge_base_loaded"] = True

        return {
            "success": True,
            "operation": "load",
            "total_files": len(document_files),
            "loaded_files": load_result.get("loaded_files", 0),
            "total_chunks": load_result.get("total_chunks", 0),
            "failed_files": load_result.get("failed_files", []),
            "summary": f"成功加载 {load_result.get('loaded_files', 0)} 个文档，共 {load_result.get('total_chunks', 0)} 个文本块"
        }

    def _query_knowledge_base(self, task: Task, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """查询知识库"""
        query = kwargs.get("query", task.get("query", ""))
        if not query:
            query = task.get("task_description", "")

        if not query:
            return {"error": "查询内容不能为空"}

        top_k = kwargs.get("top_k", 5)
        similarity_threshold = kwargs.get("similarity_threshold", 0.3)
        generate_answer = kwargs.get("generate_answer", True)
        doc_type_filter = kwargs.get("doc_type_filter")

        result = self.knowledge_base.query(
            question=query,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
            similarity_threshold=similarity_threshold,
            generate_answer=generate_answer
        )

        if result.get("error"):
            return {
                "success": False,
                "error": result["error"],
                "query": query
            }

        relevant_content = []
        for r in result.get("results", []):
            relevant_content.append({
                "content": r.get("content", ""),
                "score": r.get("score", 0),
                "source": r.get("source", ""),
                "title": r.get("title", "")
            })

        return {
            "success": True,
            "query": query,
            "results_count": len(relevant_content),
            "average_score": result.get("average_score", 0),
            "answer": result.get("answer", ""),
            "content": result.get("answer", "") if generate_answer else json.dumps(relevant_content, ensure_ascii=False,
                                                                                   indent=2),
            "summary": f"从知识库中检索到 {len(relevant_content)} 条相关结果，平均相关度: {result.get('average_score', 0):.2f}",
            "raw_data": result
        }

    def _get_knowledge_base_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        stats = self.knowledge_base.get_stats()
        return {
            "success": True,
            "operation": "stats",
            "stats": stats,
            "summary": f"知识库当前包含 {stats.get('total_vectors', 0)} 个向量"
        }


class SpiderTool(BaseWorkerTool):
    """爬虫工具 - 根据任务描述提取关键词并爬取相关信息"""

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.scraper = None
        self._initialized = False
        self._initialize_scraper()

    def _initialize_scraper(self):
        """初始化爬虫"""
        try:
            from ..tools.spider_final import WorkerScraper
            self.scraper = WorkerScraper(output_dir=self.config.SPIDER_DIR)
            self._initialized = True
            print(f"✅ {self.name} 初始化成功")
        except ImportError as e:
            print(f"❌ 爬虫工具导入失败: {e}")
            self._initialized = False
        except Exception as e:
            print(f"❌ 爬虫初始化失败: {e}")
            self._initialized = False

    def get_tool_name(self) -> str:
        return "spider_tool"

    def get_tool_description(self) -> str:
        return f"""根据任务描述提取化工领域关键词并进行网页爬取，获取最新的网络信息供参考。每个任务最多爬取 {self.config.MAX_SPIDER_RESULTS} 个网页。"""

    def get_args_schema(self) -> Type[BaseModel]:
        class SpiderArgs(BaseModel):
            search_mode: str = Field(
                default="auto",
                description="搜索模式：auto（自动提取关键词）/manual（手动指定查询）"
            )
            manual_query: Optional[str] = Field(default=None,
                                                description="手动指定的搜索查询，当search_mode=manual时需要")
            num_results: int = Field(
                default=3,
                ge=1,
                le=5,
                description=f"爬取结果数量，范围1-5（最多{self.config.MAX_SPIDER_RESULTS}个）"
            )
            use_dynamic_parse: bool = Field(default=True, description="是否使用动态解析（JavaScript渲染页面）")

        return SpiderArgs

    def execute(self, task: Task, **kwargs) -> Dict[str, Any]:
        if not self._initialized or not self.scraper:
            return {
                "error": "爬虫工具未初始化",
                "suggestion": "请检查spider_final模块是否正确安装"
            }

        if not self.config.SPIDER_ENABLED:
            return {
                "error": "爬虫功能已禁用",
                "suggestion": "请在配置中启用SPIDER_ENABLED"
            }

        try:
            task_description = task.get("task_description", "")
            task_name = task.get("task_name", "爬虫任务")

            search_mode = kwargs.get("search_mode", "auto")
            manual_query = kwargs.get("manual_query")
            num_results = min(kwargs.get("num_results", 3), self.config.MAX_SPIDER_RESULTS)
            use_dynamic_parse = kwargs.get("use_dynamic_parse", True)

            if search_mode == "manual" and manual_query:
                search_query = manual_query
            else:
                search_query = task_description

            print(f"🕷️  爬虫任务: {task_name}")
            print(f"   搜索查询: {search_query[:100]}...")

            result = self.scraper.process_chemical_task(
                task_description=search_query,
                task_name=task_name,
                num_results=num_results,
                use_dynamic_parse=use_dynamic_parse
            )

            scrape_results = result.get("scrape_results", {})
            batch_results = scrape_results.get("results", [])

            all_content = []
            successful_results = []

            for batch_result in batch_results:
                if batch_result.get("status") == "COMPLETED":
                    text_output = batch_result.get("text_output", "")
                    if text_output:
                        all_content.append(text_output)
                        successful_results.append({
                            "title": batch_result.get("section_name", "未知标题"),
                            "content_preview": text_output[:200] + "..." if len(text_output) > 200 else text_output,
                            "sources": batch_result.get("sources_used", [])
                        })

            if not all_content:
                return {
                    "success": False,
                    "error": "未能成功爬取到任何内容",
                    "raw_result": result
                }

            merged_content = "\n\n---\n\n".join(all_content)

            return {
                "success": True,
                "task_name": task_name,
                "search_query": search_query,
                "total_results": len(batch_results),
                "successful_results": len(successful_results),
                "content": merged_content,
                "summary": f"成功爬取 {len(successful_results)}/{len(batch_results)} 个网页，获取到化工相关信息",
                "results_preview": successful_results,
                "raw_data": result
            }

        except Exception as e:
            raise ToolExecutionError(f"爬虫任务失败: {str(e)}")


class CSVTool(BaseWorkerTool):
    """CSV分析工具 - 数据分析和统计"""

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        try:
            from ..tools.CSV_worker import CSVWorkerSystem
            self.csv_system = CSVWorkerSystem()
            self._initialized = True
            print(f"✅ {self.name} 初始化成功")
        except ImportError as e:
            print(f"❌ CSV工具导入失败: {e}")
            self._initialized = False

    def get_tool_name(self) -> str:
        return "csv_analysis_tool"

    def get_tool_description(self) -> str:
        return """分析CSV文件，提供数据统计、质量评估和洞察发现。支持基础统计、技术分析、统计分析和数据质量评估。"""

    def get_args_schema(self) -> Type[BaseModel]:
        class CSVArgs(BaseModel):
            analysis_type: str = Field(
                default="technical",
                description="分析类型：basic（基础统计）/technical（技术分析）/statistical（统计分析）/quality（质量评估）"
            )
            generate_table: bool = Field(
                default=True,
                description="是否生成统计表格"
            )
            target_columns: Optional[List[str]] = Field(
                default=None,
                description="指定分析的列名，为空则分析所有列"
            )

        return CSVArgs

    def get_required_resources(self) -> List[str]:
        return ["csv", "xlsx", "xls"]

    def execute(self, task: Task, **kwargs) -> Dict[str, Any]:
        if not self._initialized:
            return {
                "error": "CSV工具未初始化",
                "suggestion": "请检查CSV_worker模块是否正确安装"
            }

        try:
            resources = task.get("use_resources", [])
            csv_files = [r for r in resources if isinstance(r, str) and r.lower().endswith(('.csv', '.xlsx', '.xls'))]

            if not csv_files:
                return {
                    "error": "未找到可用的数据文件",
                    "suggestion": "请确保任务资源中包含CSV或Excel文件"
                }

            input_data = {
                "task_spec": {
                    "analysis_type": kwargs.get("analysis_type", "technical"),
                    "generate_table": kwargs.get("generate_table", True),
                    "target_columns": kwargs.get("target_columns"),
                },
                "resources": csv_files
            }

            result = self.csv_system.process_system_input(input_data)
            current_result = result.get("current_result", {})

            return {
                "success": True,
                "files_analyzed": len(csv_files),
                "tables_generated": len(current_result.get("tables", [])),
                "analysis_type": kwargs.get("analysis_type", "technical"),
                "summary": current_result.get("text_output", "数据分析完成"),
                "tables": current_result.get("tables", []),
                "raw_data": current_result
            }

        except Exception as e:
            raise ToolExecutionError(f"CSV分析失败: {str(e)}")


class ChartTool(BaseWorkerTool):
    """图表生成工具 - 数据可视化"""

    _generated_charts_cache = {}
    _cache_expiry_hours = 2

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        try:
            from ..tools.chart_generator import ChartGenerator
            self.chart_generator = ChartGenerator(
                api_key=config.API_KEY,
                base_url=config.BASE_URL,
                model=config.LLM_MODEL,
                use_image_analysis=False,
                charts_dir=config.CHARTS_DIR
            )
            self._initialized = True
            print(f"✅ {self.name} 初始化成功")
            print(f"   图表目录: {self.config.CHARTS_DIR}")

            if config.CHART_CACHE_ENABLED:
                self._load_existing_charts()
        except ImportError as e:
            print(f"❌ 图表工具导入失败: {e}")
            self._initialized = False

    def _load_existing_charts(self):
        """加载已存在的图表文件到缓存"""
        charts_dir = self.config.CHARTS_DIR
        if os.path.exists(charts_dir):
            for file_name in os.listdir(charts_dir):
                if file_name.endswith(('.png', '.svg')):
                    file_path = os.path.join(charts_dir, file_name)
                    base_name = os.path.splitext(file_name)[0]

                    chart_type = self._infer_chart_type_from_filename(file_name)

                    try:
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
                    except:
                        file_hash = base_name

                    cache_key = f"{chart_type}_{file_hash}"
                    self._generated_charts_cache[cache_key] = {
                        'path': file_path,
                        'timestamp': os.path.getmtime(file_path),
                        'chart_type': chart_type
                    }
            print(f"📂 已加载 {len(self._generated_charts_cache)} 个现有图表到缓存")

    def _infer_chart_type_from_filename(self, filename: str) -> str:
        """从文件名推断图表类型"""
        filename_lower = filename.lower()
        if 'line' in filename_lower:
            return 'line'
        elif 'bar' in filename_lower:
            return 'bar'
        elif 'scatter' in filename_lower:
            return 'scatter'
        elif 'histogram' in filename_lower or 'distribution' in filename_lower:
            return 'histogram'
        elif 'box' in filename_lower:
            return 'box'
        elif 'heatmap' in filename_lower or 'correlation' in filename_lower:
            return 'heatmap'
        elif 'pie' in filename_lower:
            return 'pie'
        else:
            return 'unknown'

    def get_tool_name(self) -> str:
        return "chart_generator_tool"

    def get_tool_description(self) -> str:
        return f"""基于数据生成专业可视化图表。支持图表类型：line/bar/scatter/histogram/box/heatmap/pie/auto。
        每个任务最多生成{self.config.MAX_CHARTS_PER_TASK}个图表，每个数据集最多生成{self.config.MAX_CHARTS_PER_DATASET}个图表。"""

    def get_args_schema(self) -> Type[BaseModel]:
        class ChartArgs(BaseModel):
            chart_type: str = Field(
                default="auto",
                description="图表类型：line/bar/scatter/histogram/box/heatmap/pie/auto"
            )
            chart_title: str = Field(default="", description="图表标题")
            x_axis: Optional[str] = Field(default=None, description="X轴字段")
            y_axis: Optional[List[str]] = Field(default=None, description="Y轴字段列表")
            color_scheme: str = Field(default="viridis", description="配色方案")
            generate_table: bool = Field(default=False, description="是否同时生成数据摘要表格")
            force_regenerate: bool = Field(default=False, description="强制重新生成图表，忽略缓存")

        return ChartArgs

    def get_required_resources(self) -> List[str]:
        return ["csv", "xlsx", "xls"]

    def execute(self, task: Task, **kwargs) -> Dict[str, Any]:
        import time

        if not self._initialized:
            return {
                "error": "图表工具未初始化",
                "suggestion": "请检查chart_generator模块是否正确安装"
            }

        if not self.config.ENABLE_CHART_GENERATION:
            return {
                "error": "图表生成功能已禁用",
                "suggestion": "请在配置中启用ENABLE_CHART_GENERATION"
            }

        try:
            resources = task.get("use_resources", [])
            data_files = [r for r in resources if isinstance(r, str) and
                          r.lower().endswith(('.csv', '.xlsx', '.xls'))]

            if not data_files:
                return {
                    "error": "未找到可用的数据文件",
                    "suggestion": "请确保任务资源中包含CSV或Excel文件"
                }

            print(f"📊 检测到 {len(data_files)} 个数据文件")

            chart_type = kwargs.get("chart_type", "auto")
            chart_title = kwargs.get("chart_title", task.get("task_name", "图表生成"))
            x_axis = kwargs.get("x_axis")
            y_axis = kwargs.get("y_axis")
            color_scheme = kwargs.get("color_scheme", "viridis")
            generate_table = kwargs.get("generate_table", task.get("generate_table", False))
            force_regenerate = kwargs.get("force_regenerate", False)

            task_desc = task.get("task_description", "")
            required_chart_types = self._parse_required_charts_from_task(task_desc, chart_type)

            max_charts_per_dataset = min(
                self.config.MAX_CHARTS_PER_DATASET,
                len(required_chart_types)
            )

            dataset_chart_types = required_chart_types[:max_charts_per_dataset]

            print(f"📊 任务分析:")
            print(f"   任务ID: {task.get('task_id')}")
            print(f"   数据文件数量: {len(data_files)}")
            print(f"   可用的图表类型: {required_chart_types}")
            print(f"   每个数据集最多生成: {max_charts_per_dataset} 个图表")
            print(f"   每个数据集将生成: {dataset_chart_types}")

            all_figures = []
            all_tables = []
            total_generated = 0
            total_existing = 0

            for dataset_idx, data_file in enumerate(data_files):
                print(f"\n📈 处理数据集 {dataset_idx + 1}/{len(data_files)}: {os.path.basename(data_file)}")

                dataset_figures, dataset_tables = self._generate_charts_for_dataset(
                    task, data_file, dataset_chart_types, chart_title,
                    x_axis, y_axis, color_scheme, generate_table,
                    force_regenerate, dataset_idx, len(data_files)
                )

                for fig in dataset_figures:
                    if fig.get('is_cached', False):
                        total_existing += 1
                    else:
                        total_generated += 1

                all_figures.extend(dataset_figures)
                all_tables.extend(dataset_tables)

                print(f"   数据集 {os.path.basename(data_file)} 生成 {len(dataset_figures)} 个图表，{len(dataset_tables)} 个表格")

            return {
                "success": True,
                "figures_generated": total_generated,
                "existing_figures": total_existing,
                "tables_generated": len(all_tables),
                "total_datasets": len(data_files),
                "chart_types_generated": list(set([f.get('chart_type', 'unknown') for f in all_figures])),
                "data_files_used": data_files,
                "summary": f"为 {len(data_files)} 个数据集生成 {total_generated} 个新图表，{len(all_tables)} 个表格，使用 {total_existing} 个缓存图表",
                "figures": all_figures,
                "tables": all_tables,
                "raw_data": {}
            }

        except Exception as e:
            print(f"❌ 图表生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ToolExecutionError(f"图表生成失败: {str(e)}")

    def _generate_charts_for_dataset(self, task: Task, data_file: str,
                                     chart_types: List[str], chart_title: str,
                                     x_axis: Optional[str], y_axis: Optional[List[str]],
                                     color_scheme: str, generate_table: bool,
                                     force_regenerate: bool,
                                     dataset_idx: int, total_datasets: int) -> tuple:
        """为单个数据集生成图表"""
        import time

        dataset_name = os.path.basename(data_file).replace('.csv', '').replace('.xlsx', '').replace('.xls', '')
        figures = []
        tables = []
        
        # 记录是否已生成过表格，避免在多图表循环中重复生成
        table_generated_for_this_dataset = False

        for chart_type_idx, chart_type in enumerate(chart_types):
            print(f"   📊 生成图表 {chart_type_idx + 1}/{len(chart_types)}: {chart_type}")

            cache_key = self._create_cache_key(data_file, chart_type, chart_title, x_axis, y_axis)

            if self.config.CHART_CACHE_ENABLED and not force_regenerate and cache_key in self._generated_charts_cache:
                cached_info = self._generated_charts_cache[cache_key]
                cache_age_hours = (time.time() - cached_info['timestamp']) / 3600

                if cache_age_hours < self._cache_expiry_hours and os.path.exists(cached_info['path']):
                    print(f"     使用缓存的图表: {os.path.basename(cached_info['path'])}")
                    figures.append({
                        'figure_id': f"cached_{dataset_name}_{chart_type}",
                        'description': f"{dataset_name} 的 {chart_type} 图表",
                        'path': cached_info['path'],
                        'chart_type': chart_type,
                        'dataset': dataset_name,
                        'is_cached': True
                    })
                    continue

            print(f"     生成新图表: {chart_type}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_id = f"{task.get('task_id', 'chart_task')}_{dataset_name}_{chart_type}_{timestamp}"

            individual_chart_title = f"{chart_title} - {dataset_name} - {chart_type}"

            # 如果需要生成表格且尚未为该数据集生成，则在第一个图表任务中同时生成表格
            should_generate_table = generate_table and not table_generated_for_this_dataset

            planner_input = {
                "messages": [],
                "tasks": [{
                    "task_id": task_id,
                    "task_name": individual_chart_title,
                    "task_description": f"为数据集 {dataset_name} 生成 {chart_type} 图表",
                    "generate_figure": True,
                    "generate_table": should_generate_table,
                    "use_resources": [data_file],
                    "chart_config": {
                        "chart_type": chart_type,
                        "x_axis": x_axis,
                        "y_axis": y_axis,
                        "color_scheme": color_scheme
                    }
                }],
                "cursor": 0
            }

            try:
                result = self.chart_generator.process_planner_input(planner_input)
                current_result = result.get("current_result", {})
                result_figures = current_result.get("figures", [])
                result_tables = current_result.get("tables", [])

                if should_generate_table and result_tables:
                    tables.extend(result_tables)
                    table_generated_for_this_dataset = True
                    print(f"     成功为数据集 {dataset_name} 生成 {len(result_tables)} 个表格")

                if result_figures:
                    for fig in result_figures:
                        if isinstance(fig, dict) and "path" in fig:
                            new_path = self._ensure_chart_in_correct_directory(fig['path'])
                            fig['path'] = new_path
                            fig['chart_type'] = chart_type
                            fig['dataset'] = dataset_name
                            fig['is_cached'] = False

                            self._generated_charts_cache[cache_key] = {
                                'path': new_path,
                                'timestamp': time.time(),
                                'chart_type': chart_type,
                                'dataset': dataset_name
                            }
                            figures.append(fig)
                else:
                    print(f"     警告: 图表生成器未返回图表")

            except Exception as e:
                print(f"     生成图表 {chart_type} 失败: {e}")
                import traceback
                traceback.print_exc()

        return figures, tables

    def _parse_required_charts_from_task(self, task_description: str, requested_type: str) -> List[str]:
        """从任务描述中解析需要生成的图表类型"""
        task_desc_lower = task_description.lower()

        if requested_type and requested_type != "auto":
            return [requested_type]

        required_charts = []

        chart_keywords = {
            "line": ["折线图", "趋势", "时序", "时间序列", "变化趋势", "line"],
            "bar": ["柱状图", "柱状", "对比", "比较", "bar", "柱形图"],
            "scatter": ["散点图", "相关性", "scatter", "散点"],
            "histogram": ["直方图", "分布", "分布图", "histogram", "distribution"],
            "box": ["箱线图", "箱型图", "异常值", "box", "boxplot", "箱线"],
            "heatmap": ["热力图", "热图", "相关性矩阵", "heatmap", "correlation"],
            "pie": ["饼图", "占比", "比例", "pie"]
        }

        for chart_type, keywords in chart_keywords.items():
            if any(keyword in task_desc_lower for keyword in keywords):
                required_charts.append(chart_type)

        if not required_charts:
            required_charts = ["line", "histogram", "scatter"]

        required_charts = list(set(required_charts))

        chart_priority = {
            "line": 1,
            "histogram": 2,
            "scatter": 3,
            "bar": 4,
            "box": 5,
            "heatmap": 6,
            "pie": 7
        }

        required_charts.sort(key=lambda x: chart_priority.get(x, 999))

        return required_charts[:self.config.MAX_CHARTS_PER_DATASET * 2]

    def _create_cache_key(self, data_file: str, chart_type: str,
                          chart_title: str, x_axis: Optional[str],
                          y_axis: Optional[List[str]]) -> str:
        """创建缓存键"""
        import hashlib

        components = []

        try:
            with open(data_file, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:8]
            components.append(file_hash)
        except:
            components.append(os.path.basename(data_file))

        components.append(chart_type)

        if chart_title:
            title_norm = chart_title.lower().replace(" ", "_").replace("-", "_")
            components.append(title_norm[:20])

        if x_axis:
            components.append(f"x_{x_axis}")
        if y_axis:
            if isinstance(y_axis, list):
                y_str = "_".join(sorted(y_axis))
            else:
                y_str = str(y_axis)
            components.append(f"y_{y_str[:30]}")

        return "_".join(components)

    def _ensure_chart_in_correct_directory(self, chart_path: str) -> str:
        """确保图表文件在正确的目录中 - 修复重复目录问题"""
        print(f"📊 处理图表文件路径: {chart_path}")
        print(f"📊 目标图表目录: {self.config.CHARTS_DIR}")

        if not os.path.exists(chart_path):
            print(f"⚠️ 图表文件不存在: {chart_path}")
            return chart_path

        file_name = os.path.basename(chart_path)
        target_path = os.path.join(self.config.CHARTS_DIR, file_name)

        print(f"📊 目标文件路径: {target_path}")

        # 如果已经在正确目录，直接返回
        if os.path.abspath(chart_path) == os.path.abspath(target_path):
            print(f"✅ 图表文件已在正确目录: {target_path}")
            return chart_path

        # 检查是否存在重复的目录结构
        if "src/agent/src/agent/charts" in chart_path.replace("\\", "/"):
            print(f"⚠️ 发现重复目录结构的图表: {chart_path}")
            print(f"   这可能是因为路径配置重复导致的")

        # 确保目标目录存在
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        try:
            # 如果目标文件已存在，先删除
            if os.path.exists(target_path):
                os.remove(target_path)

            # 移动文件到正确目录
            shutil.move(chart_path, target_path)
            print(f"✅ 移动图表文件到正确目录: {target_path}")
            return target_path
        except Exception as e:
            print(f"⚠️ 移动图表文件失败: {e}")
            # 尝试复制而不是移动
            try:
                shutil.copy2(chart_path, target_path)
                print(f"✅ 复制图表文件到正确目录: {target_path}")
                return target_path
            except Exception as e2:
                print(f"⚠️ 复制图表文件也失败: {e2}")
                return chart_path


# ==================== 5. 工具注册和管理器 ====================

class ToolManager:
    """工具管理器 - 负责工具的注册、发现和创建"""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.tool_classes: Dict[str, Type[BaseWorkerTool]] = {}
        self._register_builtin_tools()
        self._discover_custom_tools()

    def _register_builtin_tools(self):
        """注册内置工具"""
        builtin_tools = {
            "ChemicalKnowledgeBaseTool": ChemicalKnowledgeBaseTool,
            "CSVTool": CSVTool,
            "ChartTool": ChartTool,
            "SpiderTool": SpiderTool
        }

        for tool_name, tool_class in builtin_tools.items():
            if tool_name in self.config.ENABLED_TOOLS:
                self.register_tool(tool_name, tool_class)
                print(f"✅ 注册内置工具类: {tool_name}")

    def _discover_custom_tools(self):
        """发现和注册自定义工具"""
        tools_dir = self.config.TOOLS_DIR
        if not os.path.exists(tools_dir):
            os.makedirs(tools_dir, exist_ok=True)
            return

        for file_name in os.listdir(tools_dir):
            if file_name.endswith('.py') and not file_name.startswith('_'):
                try:
                    module_name = file_name[:-3]
                    module_path = os.path.join(tools_dir, file_name)

                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (inspect.isclass(attr) and
                                issubclass(attr, BaseWorkerTool) and
                                attr != BaseWorkerTool):

                            try:
                                tool_instance = attr(self.config)
                                tool_name = tool_instance.name
                                self.register_tool(tool_name, attr)
                                print(f"✅ 发现自定义工具: {tool_name}")
                            except:
                                self.register_tool(attr_name, attr)
                                print(f"✅ 发现自定义工具: {attr_name}")

                except Exception as e:
                    print(f"⚠️ 加载工具文件 {file_name} 失败: {e}")

    def register_tool(self, tool_name: str, tool_class: Type[BaseWorkerTool]):
        """注册工具类"""
        self.tool_classes[tool_name] = tool_class

    def get_tool_instance(self, tool_name: str) -> Optional[BaseWorkerTool]:
        """获取工具实例"""
        if tool_name in self.tool_classes:
            try:
                return self.tool_classes[tool_name](self.config)
            except Exception as e:
                print(f"⚠️ 实例化工具 {tool_name} 失败: {e}")
                return None
        return None

    def get_available_tools_for_task(self, task: Task) -> List[BaseWorkerTool]:
        """根据任务获取可用工具列表"""
        available_tools = []

        tool_requirements = task.get("tool_requirements", [])
        if not tool_requirements:
            if task.get("use_rag"):
                tool_requirements.append("chemical_knowledge_base_tool")
            if task.get("generate_table"):
                tool_requirements.append("csv_analysis_tool")
            if task.get("generate_figure"):
                tool_requirements.append("chart_generator_tool")

        for tool_name in self.config.ENABLED_TOOLS:
            tool_class = self.tool_classes.get(tool_name)
            if tool_class:
                tool_instance = tool_class(self.config)
                if tool_instance.name not in [t.name for t in available_tools]:
                    available_tools.append(tool_instance)
        print(f"过滤前可用工具：{[tool.name for tool in available_tools]}")

        filtered_tools = []
        for tool in available_tools:
            if (not tool_requirements or
                    tool.name in tool_requirements or
                    any(req in tool.name for req in tool_requirements)):

                if tool.validate_task(task):
                    filtered_tools.append(tool)

        return filtered_tools

    def create_langchain_tools(self, task: Task) -> List[BaseTool]:
        """创建LangChain兼容的工具列表"""
        available_tools = self.get_available_tools_for_task(task)
        print(f"可用工具：{[tool.name for tool in available_tools]}")
        langchain_tools = []

        for tool in available_tools:
            try:
                langchain_tool = tool.create_langchain_tool(task)
                langchain_tools.append(langchain_tool)
                print(f"🛠️ 加载工具: {tool.name}")

            except Exception as e:
                print(f"⚠️ 创建LangChain工具 {tool.name} 失败: {e}")
                import traceback
                traceback.print_exc()

        return langchain_tools


# ==================== 6. 大模型自主工具调用节点 ====================

class AutonomousToolNode:
    """大模型自主工具调用节点"""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.tool_manager = ToolManager(config)

        # 统一使用 get_llm
        llm_config = {
            "configurable": {
                "model_name": self.config.LLM_MODEL,
                "temperature": self.config.TEMPERATURE,
                "base_url": self.config.BASE_URL
            }
        }
        # Worker节点需要自主调用工具并生成文本报告，不强制JSON模式
        self.llm_client = get_llm(llm_config, json_mode=False)

    def process(self, state: State) -> Dict[str, Any]:
        """处理任务：让大模型自主调用工具"""
        tasks = state.get("tasks", [])
        cursor = state.get("cursor", 0)

        if cursor >= len(tasks):
            return {"worker_state": {"next_node": "generate_final_result_node"}}

        current_task = tasks[cursor]
        task_name = current_task.get("task_name", f"任务{cursor + 1}")

        print(f"\n{'=' * 60}")
        print(f"🎯 开始处理任务 {cursor + 1}/{len(tasks)}: {task_name}")
        print(f"📋 任务ID: {current_task.get('task_id')}")
        print(f"{'=' * 60}")

        try:
            start_time = datetime.now()

            tools = self.tool_manager.create_langchain_tools(current_task)

            if not tools:
                print("⚠️ 没有可用的工具，跳过此任务")
                return self._create_skip_result(current_task, cursor, "没有可用的工具")

            print(f"📦 可用工具: {[tool.name for tool in tools]}")

            if self.config.AUTO_LOAD_RESOURCES and not current_task.get("knowledge_base_loaded", False):
                self._auto_load_to_knowledge_base(current_task, tools)

            system_prompt = self._build_system_prompt(current_task, tools)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=self._build_task_prompt(current_task))
            ]

            try:
                llm_with_tools = self.llm_client.bind_tools(tools)
            except Exception as e:
                print(f"⚠️ 绑定工具到LLM失败: {e}")
                llm_with_tools = self.llm_client

            final_content, tool_calls, tool_usage_stats = self._execute_tool_loop(
                llm_with_tools, messages, tools, current_task
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            knowledge_base_used = any("knowledge_base" in call["tool"] for call in tool_calls)
            spider_results_used = any("spider" in call["tool"] for call in tool_calls)

            task_result = self._create_task_result(
                current_task, cursor, final_content, tool_calls,
                tool_usage_stats, execution_time, knowledge_base_used, spider_results_used
            )

            all_results = state.get("all_results", []).copy()
            all_results.append(task_result)

            worker_state = state.get("worker_state", {}).copy()
            worker_state.update({
                "next_node": "generate_task_result_node",
                "current_section": task_name,
                "progress": (cursor + 1) / len(tasks),
                "tool_usage_stats": tool_usage_stats,
                "last_task_time": execution_time,
                "knowledge_base_used": knowledge_base_used,
                "spider_results_used": spider_results_used
            })

            tool_execution_history = state.get("tool_execution_history", []).copy()
            tool_execution_history.append({
                "task_id": current_task.get("task_id"),
                "task_name": task_name,
                "tool_calls": tool_calls,
                "tool_usage_stats": tool_usage_stats,
                "execution_time": execution_time,
                "knowledge_base_used": knowledge_base_used,
                "spider_results_used": spider_results_used,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "current_task": current_task,
                "current_result": task_result,
                "cursor": cursor,
                "all_results": all_results,
                "worker_state": worker_state,
                "tool_execution_history": tool_execution_history,
                "available_tools": [{"name": tool.name, "description": tool.description} for tool in tools]
            }

        except Exception as e:
            print(f"❌ 任务执行失败: {e}")
            import traceback
            traceback.print_exc()

            task_result = self._create_error_result(current_task, cursor, str(e))

            all_results = state.get("all_results", []).copy()
            all_results.append(task_result)

            worker_state = state.get("worker_state", {}).copy()
            worker_state.update({
                "next_node": "generate_task_result_node",
                "current_section": task_name,
                "progress": (cursor + 1) / len(tasks)
            })

            return {
                "current_task": current_task,
                "current_result": task_result,
                "cursor": cursor,
                "all_results": all_results,
                "worker_state": worker_state
            }

    def _auto_load_to_knowledge_base(self, task: Task, tools: List[BaseTool]):
        """自动将任务资源中的文档加载到知识库"""
        kb_tool = None
        for tool in tools:
            if "knowledge_base" in tool.name:
                kb_tool = tool
                break

        if not kb_tool:
            return

        resources = task.get("use_resources", [])
        supported_extensions = ['.txt', '.pdf', '.docx', '.md', '.csv', '.xlsx', '.xls']
        document_files = [
            r for r in resources
            if isinstance(r, str) and os.path.splitext(r.lower())[1] in supported_extensions
        ]

        if not document_files:
            return

        print(f"📚 自动加载 {len(document_files)} 个文档到知识库...")

        try:
            load_result = kb_tool.invoke({
                "operation": "load"
            })

            if "success" in load_result and load_result["success"]:
                print(f"✅ 成功加载 {load_result.get('loaded_files', 0)} 个文档到知识库")
                task["knowledge_base_loaded"] = True
            else:
                print(f"⚠️ 自动加载文档失败: {load_result.get('error', '未知错误')}")

        except Exception as e:
            print(f"⚠️ 自动加载文档到知识库时出错: {e}")

    def _build_system_prompt(self, task: Task, tools: List[BaseTool]) -> str:
        """构建系统提示词 - 明确要求生成报告正文"""
        task_name = task.get("task_name", "")
        task_desc = task.get("task_description", "")
        resources = task.get("use_resources", [])
        task_type = task.get("task_type", "")

        tool_descriptions = []
        for tool in tools:
            tool_info = f"- {tool.name}: {tool.description}"
            tool_descriptions.append(tool_info)
            
        resources_str = ', '.join([str(r) for r in resources[:5]])
        if len(resources) > 5:
            resources_str += '...（更多资源）'

        # Load template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "../../../prompts/worker_system_template.md")
        
        with open(template_path, "r", encoding="utf-8") as f:
            template_str = f.read()
        
        prompt_template = PromptTemplate.from_template(template_str)
        prompt = prompt_template.format(
            task_name=task_name,
            task_type=task_type,
            task_desc=task_desc,
            resources_str=resources_str,
            tool_descriptions="\n".join(tool_descriptions),
            max_spider_results=getattr(self.config, 'MAX_SPIDER_RESULTS', 10),
            max_charts_per_task=getattr(self.config, 'MAX_CHARTS_PER_TASK', 6),
            max_charts_per_dataset=getattr(self.config, 'MAX_CHARTS_PER_DATASET', 6)
        )

        # Append conditional warnings
        if not getattr(self.config, 'ENABLE_CHART_GENERATION', True):
            prompt += "\n\n⚠️ 警告：图表生成功能已禁用，请不要调用图表生成工具。"

        if not getattr(self.config, 'SPIDER_ENABLED', True):
            prompt += "\n\n⚠️ 警告：爬虫功能已禁用，请不要调用爬虫工具。"

        return prompt

    def _build_task_prompt(self, task: Task) -> str:
        """构建任务提示词"""
        task_desc = task.get("task_description", "")
        query = task.get("query", "")
        
        # 构建查询部分
        query_section = ""
        if query:
            query_section = f"相关查询：{query}"
        
        # 构建资源部分
        resources = task.get("use_resources", [])
        data_files = [r for r in resources if isinstance(r, str) and r.lower().endswith(('.csv', '.xlsx', '.xls'))]
        
        data_files_section = ""
        if data_files:
            lines = [f"可用数据文件 ({len(data_files)} 个):"]
            for i, file in enumerate(data_files[:5]):
                lines.append(f"  {i + 1}. {os.path.basename(file)}")
            if len(data_files) > 5:
                lines.append(f"  ... 还有 {len(data_files) - 5} 个文件")
            data_files_section = "\n".join(lines)
                
        # Load template from file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "../../../prompts/worker_task_template.md")
        
        with open(template_path, "r", encoding="utf-8") as f:
            template_str = f.read()
        
        # 使用 ChatPromptTemplate 格式化
        prompt_template = ChatPromptTemplate.from_template(template_str)
        
        # 获取 MAX_CHARTS_PER_TASK，默认为 6
        max_charts = getattr(self.config, 'MAX_CHARTS_PER_TASK', 6)
        
        messages = prompt_template.format_messages(
            task_description=task_desc,
            query_section=query_section,
            data_files_section=data_files_section,
            max_charts=max_charts
        )
        
        # 返回消息内容
        return messages[0].content

    def _extract_tool_args(self, tool_call):
        """提取工具调用参数 - 修复字符串格式问题"""
        try:
            print(f"      🔍 开始提取工具参数")
            print(f"      🔍 tool_call类型: {type(tool_call)}")

            if isinstance(tool_call, dict):
                print(f"      🔍 tool_call是字典，键: {list(tool_call.keys())}")

                for key in ['args', 'tool_input', 'input', 'arguments', 'parameters', 'args_schema']:
                    if key in tool_call and tool_call[key]:
                        args = tool_call[key]
                        print(f"      🔍 在键 {key} 中找到参数，类型: {type(args)}")

                        if isinstance(args, dict):
                            return args
                        elif isinstance(args, str):
                            return self._parse_string_args(args)

                if 'tool_calls' in tool_call:
                    return self._extract_tool_args(tool_call['tool_calls'][0])

            elif hasattr(tool_call, '__dict__'):
                tool_call_dict = tool_call.__dict__
                print(f"      🔍 tool_call有__dict__，键: {list(tool_call_dict.keys())}")

                for key in ['args', 'tool_input', 'input', 'arguments', 'parameters', 'args_schema']:
                    if key in tool_call_dict and tool_call_dict[key]:
                        args = tool_call_dict[key]
                        print(f"      🔍 在__dict__中找到键 {key}, 类型: {type(args)}")

                        if isinstance(args, dict):
                            return args
                        elif isinstance(args, str):
                            return self._parse_string_args(args)

            for attr_name in ['args', 'tool_input', 'input', 'arguments', 'parameters']:
                if hasattr(tool_call, attr_name):
                    args = getattr(tool_call, attr_name)
                    if args:
                        print(f"      🔍 在属性 {attr_name} 中找到参数，类型: {type(args)}")

                        if isinstance(args, dict):
                            return args
                        elif isinstance(args, str):
                            return self._parse_string_args(args)

            print(f"      ⚠️ 无法提取参数，返回空字典")
            return {}

        except Exception as e:
            print(f"      ❌ 提取参数时出错: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _parse_string_args(self, args_str: str) -> Dict[str, Any]:
        """解析字符串参数为字典"""
        if not args_str or not isinstance(args_str, str):
            return {}

        cleaned_str = args_str.strip()
        print(f"      🔍 解析字符串参数: {cleaned_str[:100]}...")

        try:
            result = json.loads(cleaned_str)
            print(f"      ✅ 直接JSON解析成功")
            return result
        except json.JSONDecodeError:
            print(f"      ⚠️ JSON解析失败，尝试其他方法")

        try:
            fixed_str = cleaned_str.replace("'", '"')
            fixed_str = re.sub(r'(\s*{\s*|\s*,\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed_str)
            result = json.loads(fixed_str)
            print(f"      ✅ 修复后JSON解析成功")
            return result
        except (json.JSONDecodeError, Exception) as e:
            print(f"      ⚠️ 修复JSON解析失败: {e}")

        try:
            result = ast.literal_eval(cleaned_str)
            if isinstance(result, dict):
                print(f"      ✅ ast.literal_eval解析成功")
                return result
        except (SyntaxError, ValueError) as e:
            print(f"      ⚠️ ast.literal_eval解析失败: {e}")

        if len(cleaned_str) < 100 and not cleaned_str.startswith('{'):
            print(f"      ⚠️ 返回查询字符串作为参数")
            return {"query": cleaned_str}

        try:
            start = cleaned_str.find('{')
            end = cleaned_str.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = cleaned_str[start:end + 1]
                result = json.loads(json_str)
                print(f"      ✅ 提取JSON部分解析成功")
                return result
        except:
            pass

        print(f"      ❌ 所有解析方法都失败，返回空字典")
        return {}

    def _execute_tool_loop(self, llm_with_tools, initial_messages: List,
                           tools: List[BaseTool], task: Task) -> tuple:
        """执行工具调用循环"""
        messages = initial_messages.copy()
        tool_calls = []
        tool_usage_stats = {}
        max_iterations = self.config.MAX_TOOL_ITERATIONS
        tool_map = {tool.name: tool for tool in tools}

        for tool in tools:
            tool_usage_stats[tool.name] = 0

        generated_chart_types = set()

        print(f"🔄 开始工具调用循环，最多{max_iterations}次迭代")

        for iteration in range(max_iterations):
            print(f"\n  🔄 迭代 {iteration + 1}/{max_iterations}")

            try:
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                if not hasattr(response, 'tool_calls') or not response.tool_calls:
                    print(f"    ✅ 没有更多工具调用，任务完成")

                    content = response.content
                    if content and len(content.strip()) > 200:
                        print(f"    ✅ 生成报告正文，长度: {len(content)} 字符")
                        return content, tool_calls, tool_usage_stats
                    else:
                        print(f"    ⚠️ 生成的内容太短，尝试生成更详细的报告")
                        messages.append(HumanMessage(
                            content="请生成一个详细的分析报告正文，包含数据分析结果、关键发现、结论和建议。"
                        ))
                        final_response = self.llm_client.invoke(messages)
                        return final_response.content, tool_calls, tool_usage_stats

                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name")

                    if "chart" in tool_name.lower():
                        tool_args = self._extract_tool_args(tool_call)
                        chart_type = tool_args.get("chart_type", "unknown")

                        if chart_type in generated_chart_types:
                            print(f"    ⚠️ 跳过重复的图表类型: {chart_type}")
                            continue

                        if len(generated_chart_types) >= self.config.MAX_CHARTS_PER_TASK:
                            print(f"    ⚠️ 已达到最大图表数限制({self.config.MAX_CHARTS_PER_TASK})，跳过图表生成")
                            continue

                        generated_chart_types.add(chart_type)

                    print(f"    🔧 调用工具: {tool_name}")

                    try:
                        tool_args = self._extract_tool_args(tool_call)

                        if tool_args:
                            print(f"    🔍 工具参数: {json.dumps(tool_args, ensure_ascii=False, indent=4)}")
                        else:
                            print(f"    ⚠️ 无法获取工具参数，使用默认参数")

                        tool_function = tool_map.get(tool_name)
                        if not tool_function:
                            raise ValueError(f"未知工具: {tool_name}")

                        tool_result = tool_function.invoke(tool_args)

                        # 尝试解析结果以保存完整数据
                        full_result = tool_result
                        try:
                            if isinstance(tool_result, str) and (tool_result.strip().startswith('{') or tool_result.strip().startswith('[')):
                                full_result = json.loads(tool_result)
                        except:
                            pass

                        tool_calls.append({
                            "tool": tool_name,
                            "parameters": tool_args,
                            "result": str(tool_result)[:500] + "..." if len(str(tool_result)) > 500 else str(
                                tool_result),
                            "full_result": full_result,
                            "success": True,
                            "iteration": iteration + 1,
                            "timestamp": datetime.now().isoformat()
                        })

                        tool_usage_stats[tool_name] = tool_usage_stats.get(tool_name, 0) + 1

                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call.get("id", f"call_{iteration}_{len(tool_calls)}")
                        ))

                        print(f"      ✅ 工具执行成功")

                    except Exception as e:
                        print(f"      ❌ 工具调用失败: {e}")
                        import traceback
                        traceback.print_exc()

                        tool_calls.append({
                            "tool": tool_name,
                            "parameters": tool_args,
                            "result": f"工具调用失败: {str(e)}",
                            "success": False,
                            "iteration": iteration + 1,
                            "timestamp": datetime.now().isoformat()
                        })

                        messages.append(ToolMessage(
                            content=f"工具调用失败: {str(e)}",
                            tool_call_id=tool_call.get("id", f"call_{iteration}_{len(tool_calls)}")
                        ))

            except Exception as e:
                print(f"    ⚠️ 迭代 {iteration + 1} 发生异常: {e}")
                import traceback
                traceback.print_exc()

        print("⚠️ 达到最大迭代次数，生成最终回答")
        messages.append(HumanMessage(
            content="请基于所有工具调用的结果，生成一个详细的分析报告正文。"
        ))
        final_response = self.llm_client.invoke(messages)
        return final_response.content, tool_calls, tool_usage_stats

    def _create_task_result(self, task: Task, cursor: int, content: str,
                            tool_calls: List[Dict], tool_usage_stats: Dict,
                            execution_time: float, knowledge_base_used: bool,
                            spider_results_used: bool) -> TaskResult:
        """创建任务结果 - 确保使用大模型生成的内容作为报告正文"""
        figures_generated = 0
        figures = []
        tables = []
        for call in tool_calls:
            # 提取图表
            if "chart" in call["tool"].lower() and call["success"]:
                # 优先从 full_result 获取
                if "full_result" in call:
                    result_data = call["full_result"]
                    if isinstance(result_data, dict) and "figures" in result_data:
                        figures.extend(result_data["figures"])
                        figures_generated += len(result_data["figures"])
                    
                    # 同时尝试从 full_result 提取表格
                    if isinstance(result_data, dict) and "tables" in result_data:
                        tables.extend(result_data["tables"])
                    
                    if "full_result" in call and "figures" in result_data:
                        continue

                try:
                    result_str = call.get("result", "")
                    if result_str and "figures" in result_str:
                        result_data = json.loads(result_str)
                        if "figures" in result_data:
                            figures.extend(result_data["figures"])
                            figures_generated += len(result_data["figures"])
                        if "tables" in result_data:
                            tables.extend(result_data["tables"])
                except:
                    figures_generated += 1
            
            # 提取表格 (不仅限于图表工具，如 csv_analysis_tool 也会生成表格)
            elif call["success"]:
                if "full_result" in call:
                    result_data = call["full_result"]
                    if isinstance(result_data, dict) and "tables" in result_data:
                        tables.extend(result_data["tables"])
                else:
                    try:
                        result_str = call.get("result", "")
                        if result_str and "tables" in result_str:
                            result_data = json.loads(result_str)
                            if "tables" in result_data:
                                tables.extend(result_data["tables"])
                    except:
                        pass

        if not content or len(content.strip()) < 100:
            content = self._extract_content_from_tool_calls(tool_calls)

        content = self._clean_report_content(content)

        return {
            "task_id": task.get("task_id", f"task_{cursor}"),
            "section_name": task.get("task_name", f"任务{cursor + 1}"),
            "text_output": content,
            "status": "COMPLETED",
            "tables": tables,
            "figures": figures,
            "sources_used": task.get("use_resources", []),
            "figures_generated": figures_generated,
            "word_count": len(content),
            "generated_at": datetime.now().isoformat(),
            "execution_time": execution_time,
            "tool_calls": tool_calls,
            "tool_usage_stats": tool_usage_stats,
            "knowledge_base_used": knowledge_base_used,
            "spider_results_used": spider_results_used,
            "error": None
        }

    def _extract_content_from_tool_calls(self, tool_calls: List[Dict]) -> str:
        """从工具调用结果中提取内容"""
        content_parts = []

        for call in tool_calls:
            if call.get("success") and call.get("result"):
                result_str = call.get("result", "")
                if len(result_str) > 100:
                    content_parts.append(f"【{call.get('tool', '工具')}分析结果】:")

                    try:
                        result_data = json.loads(result_str)
                        if isinstance(result_data, dict):
                            if "content" in result_data:
                                content_parts.append(str(result_data["content"]))
                            elif "summary" in result_data:
                                content_parts.append(str(result_data["summary"]))
                            elif "text_output" in result_data:
                                content_parts.append(str(result_data["text_output"]))
                            elif "answer" in result_data:
                                content_parts.append(str(result_data["answer"]))
                        else:
                            content_parts.append(result_str[:500])
                    except:
                        content_parts.append(result_str[:500])

        return "\n\n".join(content_parts) if content_parts else "分析完成，但未能生成详细报告正文。"

    def _clean_report_content(self, content: str) -> str:
        """清理报告内容，移除工具调用细节等无关信息"""
        if not content:
            return content

        patterns_to_remove = [
            r'工具调用.*?:',
            r'调用工具.*?:',
            r'【.*?工具.*?】',
            r'Tool.*?:',
            r'参数.*?:',
            r'Arguments.*?:',
            r'执行.*?:',
            r'结果.*?:',
            r'Result.*?:',
        ]

        cleaned_content = content
        for pattern in patterns_to_remove:
            cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE)

        cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)

        return cleaned_content.strip()

    def _create_error_result(self, task: Task, cursor: int, error: str) -> TaskResult:
        """创建错误结果"""
        return {
            "task_id": task.get("task_id", f"task_{cursor}"),
            "section_name": task.get("task_name", f"任务{cursor + 1}"),
            "text_output": f"任务执行失败: {error}",
            "status": "FAILED",
            "tables": [],
            "figures": [],
            "sources_used": task.get("use_resources", []),
            "figures_generated": 0,
            "word_count": 0,
            "generated_at": datetime.now().isoformat(),
            "execution_time": 0.0,
            "tool_calls": [],
            "tool_usage_stats": {},
            "knowledge_base_used": False,
            "spider_results_used": False,
            "error": error
        }

    def _create_skip_result(self, task: Task, cursor: int, reason: str) -> Dict[str, Any]:
        """创建跳过结果"""
        task_result = {
            "task_id": task.get("task_id", f"task_{cursor}"),
            "section_name": task.get("task_name", f"任务{cursor + 1}"),
            "text_output": f"任务被跳过: {reason}",
            "status": "SKIPPED",
            "tables": [],
            "figures": [],
            "sources_used": task.get("use_resources", []),
            "figures_generated": 0,
            "word_count": 0,
            "generated_at": datetime.now().isoformat(),
            "execution_time": 0.0,
            "tool_calls": [],
            "tool_usage_stats": {},
            "knowledge_base_used": False,
            "spider_results_used": False,
            "error": reason
        }

        return {
            "current_task": task,
            "current_result": task_result,
            "cursor": cursor,
            "all_results": [task_result],
            "worker_state": {
                "next_node": "generate_task_result_node",
                "current_section": task.get("task_name"),
                "progress": 1.0
            }
        }


# ==================== 7. 节点函数 ====================

def autonomous_tool_node(state: State) -> Dict[str, Any]:
    """大模型自主工具调用节点（包装器）"""
    # 获取会话隔离的缓存目录
    session_cache_dir = get_session_cache_dir(state)
    config = WorkerConfig(BASE_DIR=session_cache_dir)
    node = AutonomousToolNode(config)
    return node.process(state)


def generate_task_result_node(state: State) -> Dict[str, Any]:
    """生成任务结果节点"""
    worker_state = state.get("worker_state", {}).copy()
    current_result = state.get("current_result")

    output = {
        "from": "Worker",
        "to": "Planner",
        "type": "TASK_RESULT",
        "section": worker_state.get("current_section", ""),
        "progress": worker_state.get("progress", 0.0),
        "task_id": current_result.get("task_id") if current_result else None,
        "content": {
            "current_result": current_result,
            "completed_sections": [worker_state.get("current_section", "")],
            "tool_usage": worker_state.get("tool_usage_stats", {}),
            "execution_time": worker_state.get("last_task_time", 0.0),
            "knowledge_base_used": worker_state.get("knowledge_base_used", False),
            "spider_results_used": worker_state.get("spider_results_used", False)
        }
    }

    messages = state.get("messages", []).copy()
    messages.append(AIMessage(content=json.dumps(output, ensure_ascii=False)))

    # 任务完成后，严格跳转到 end 节点，交由 Verifier 验证
    worker_state["next_node"] = "end"

    return {
        "messages": messages,
        "worker_state": worker_state
    }


def generate_final_result_node(state: State) -> Dict[str, Any]:
    """生成最终结果节点"""
    all_results = state.get("all_results", [])
    
    # 获取会话隔离的缓存目录
    session_cache_dir = get_session_cache_dir(state)
    config = WorkerConfig(BASE_DIR=session_cache_dir)

    report_paths = _generate_single_reports(all_results, config)

    output = {
        "from": "Worker",
        "to": "Planner",
        "type": "FINAL_RESULT",
        "content": {
            "all_results": all_results,
            "report_paths": report_paths
        }
    }

    messages = state.get("messages", []).copy()
    messages.append(AIMessage(content=json.dumps(output, ensure_ascii=False)))

    print(f"\n{'=' * 60}")
    print("✅ 所有任务处理完成")
    print(f"   报告已保存到: {config.REPORT_DIR}")
    print(f"   图表已保存到: {config.CHARTS_DIR}")
    print(f"   知识库路径: {config.KNOWLEDGE_BASE_DIR}")
    print(f"{'=' * 60}")

    return {
        "messages": messages,
        "worker_state": {"next_node": "end"}
    }


def _generate_single_reports(all_results: List[TaskResult], config: WorkerConfig) -> Dict[str, str]:
    """生成报告 - 每个任务只生成一个报告"""
    report_paths = {}

    if config.GENERATE_SINGLE_REPORT:
        for i, result in enumerate(all_results):
            task_id = result.get("task_id", f"task_{i}")
            task_name = result.get("section_name", f"任务{i + 1}").replace("/", "_").replace("\\", "_")

            import re
            task_name_clean = re.sub(r'[<>:"/\\|?*]', '_', task_name)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{task_name_clean}_{timestamp}.md"
            task_report_path = os.path.join(config.REPORT_DIR, report_filename)

            _generate_task_report_fixed(result, task_report_path, config)
            report_paths[task_id] = task_report_path

            print(f"📄 生成任务报告: {task_report_path}")
    else:
        report_paths = _generate_clean_reports(all_results, config)

    return report_paths


def _generate_clean_reports(all_results: List[TaskResult], config: WorkerConfig) -> Dict[str, str]:
    """生成只包含正文内容的报告（备份方法）"""
    report_paths = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_report_path = os.path.join(config.REPORT_DIR, f"final_report_{timestamp}.md")
    _generate_clean_summary_report(all_results, summary_report_path)
    report_paths["final_report"] = summary_report_path

    for i, result in enumerate(all_results):
        task_id = result.get("task_id", f"task_{i}")
        task_report_path = os.path.join(config.REPORT_DIR, f"{task_id}_{timestamp}.md")
        _generate_clean_task_report(result, task_report_path)
        report_paths[f"task_{task_id}"] = task_report_path

    return report_paths


def _generate_task_report_fixed(task_result: TaskResult, output_path: str, config: WorkerConfig) -> str:
    """修复后的报告生成函数 - 只包含报告正文"""
    task_name = task_result.get("section_name", "任务报告")
    text_output = task_result.get("text_output", "")

    if not text_output or len(text_output.strip()) < 100:
        tool_calls = task_result.get("tool_calls", [])
        if tool_calls:
            for call in tool_calls:
                if call.get("success") and call.get("result"):
                    result_str = call.get("result", "")
                    if result_str and len(result_str) > 50:
                        text_output = f"工具调用结果:\n{result_str[:500]}..."
                        break

    figures = task_result.get("figures", [])
    chart_references = ""

    if figures:
        chart_references = "\n\n## 生成的图表\n\n"

        dataset_charts = {}
        for fig in figures:
            if isinstance(fig, dict):
                dataset = fig.get('dataset', '未知数据集')
                if dataset not in dataset_charts:
                    dataset_charts[dataset] = []
                dataset_charts[dataset].append(fig)

        for dataset, charts in dataset_charts.items():
            chart_references += f"### {dataset}\n\n"
            for i, fig in enumerate(charts):
                if "path" in fig:
                    fig_path = fig.get("path", "")
                    fig_desc = fig.get("description", f"{fig.get('chart_type', '未知类型')}图表")
                    is_cached = fig.get('is_cached', False)

                    if os.path.exists(fig_path):
                        fig_filename = os.path.basename(fig_path)
                        fig_path_display = f"../charts/{fig_filename}"

                        cache_note = " (缓存)" if is_cached else ""
                        chart_references += f"{i + 1}. {fig_desc}{cache_note}: ![图表]({fig_path_display})\n"
                    else:
                        chart_references += f"{i + 1}. {fig_desc}\n"
            chart_references += "\n"

    report_content = f"""# {task_name}

{text_output}

{chart_references}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return output_path


def _generate_clean_summary_report(all_results: List[TaskResult], output_path: str):
    """生成只包含正文的综合报告"""
    report_content = ""

    for i, result in enumerate(all_results):
        task_name = result.get("section_name", f"任务{i + 1}")
        text_output = result.get("text_output", "")

        report_content += f"# {task_name}\n\n"
        report_content += f"{text_output}\n\n"
        report_content += "---\n\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return output_path


def _generate_clean_task_report(task_result: TaskResult, output_path: str):
    """生成只包含正文的单个任务报告（备份方法）"""
    task_name = task_result.get("section_name", "任务报告")
    text_output = task_result.get("text_output", "")

    report_content = f"# {task_name}\n\n{text_output}"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return output_path


def router_node(state: State) -> Dict[str, Any]:
    """路由节点 - 决定是执行下一个任务还是生成最终结果"""
    tasks = state.get("tasks", [])
    cursor = state.get("cursor", 0)

    # 1. 如果 state 中已经有任务列表
    if tasks:
        if cursor < len(tasks):
            print(f"📥 继续执行任务 {cursor + 1}/{len(tasks)}")
            return {"worker_state": {"next_node": "autonomous_tool_node"}}
        else:
            print(f"📥 所有任务已执行完毕，准备生成最终结果报告")
            return {"worker_state": {"next_node": "generate_final_result_node"}}

    # 2. 如果没有任务列表，尝试从最新消息中提取
    if not state.get("messages"):
        return {"worker_state": {"next_node": "end"}}

    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        return {"worker_state": {"next_node": "end"}}

    try:
        message_data = json.loads(last_message.content)
        msg_type = message_data.get("type", "")

        if msg_type == "PLAN_RESULT":
            content = message_data.get("content", {})
            tasks = content.get("tasks", [])

            if tasks:
                print(f"📥 收到计划结果，包含 {len(tasks)} 个任务")
                return {
                    "tasks": tasks,
                    "cursor": 0,
                    "worker_state": {"next_node": "autonomous_tool_node"}
                }

        return {"worker_state": {"next_node": "end"}}

    except json.JSONDecodeError:
        return {"worker_state": {"next_node": "end"}}


def route_decision(state: State) -> str:
    """条件路由函数"""
    return state.get("worker_state", {}).get("next_node", "end")


def create_worker_workflow():
    """创建工作流"""
    workflow = StateGraph(State)

    workflow.add_node("router_node", router_node)
    workflow.add_node("autonomous_tool_node", autonomous_tool_node)
    workflow.add_node("generate_task_result_node", generate_task_result_node)
    workflow.add_node("generate_final_result_node", generate_final_result_node)

    workflow.set_entry_point("router_node")

    workflow.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "autonomous_tool_node": "autonomous_tool_node",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "autonomous_tool_node",
        route_decision,
        {
            "generate_task_result_node": "generate_task_result_node",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "generate_task_result_node",
        route_decision,
        {
            "autonomous_tool_node": "autonomous_tool_node",
            "generate_final_result_node": "generate_final_result_node",
            "end": END
        }
    )

    workflow.add_edge("generate_final_result_node", END)

    return workflow.compile()


# ==================== 8. Worker智能体 ====================

class WorkerAgent:
    """Worker智能体"""

    def __init__(self, config=None):
        self.config = config or WorkerConfig()
        self.workflow = create_worker_workflow()
        self._ensure_directories()
        self._check_and_fix_duplicate_dirs()  # 添加目录检查和修复
        self._cleanup_old_charts()

    def _ensure_directories(self):
        """确保目录存在"""
        print("\n📁 正在创建目录结构...")
        print(f"   当前文件位置: {__file__}")
        print(f"   BASE_DIR: {self.config.BASE_DIR}")
        print(f"   CHARTS_DIR: {self.config.CHARTS_DIR}")
        print(f"   REPORT_DIR: {self.config.REPORT_DIR}")

        charts_dir = self.config.CHARTS_DIR
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir, exist_ok=True)
            print(f"✅ 创建图表目录: {charts_dir}")
        else:
            print(f"✅ 图表目录已存在: {charts_dir}")

        for dir_path in [self.config.REPORT_DIR, self.config.OUTPUT_DIR,
                         self.config.LOGS_DIR, self.config.KNOWLEDGE_BASE_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ 创建目录: {dir_path}")

    def _check_and_fix_duplicate_dirs(self):
        """检查并修复重复目录结构"""
        base_agent_dir = self.config.BASE_DIR  # E:\pycharm\RAG_Agent\src\agent

        # 检查是否存在重复的 src/agent 结构
        duplicate_charts_dir = os.path.join(base_agent_dir, "src", "agent", "charts")
        duplicate_report_dir = os.path.join(base_agent_dir, "src", "agent", "report")

        # 如果存在重复目录，将其内容移动到正确位置
        for duplicate_dir, correct_dir in [
            (duplicate_charts_dir, self.config.CHARTS_DIR),
            (duplicate_report_dir, self.config.REPORT_DIR)
        ]:
            if os.path.exists(duplicate_dir) and os.path.exists(correct_dir):
                print(f"🔄 发现重复目录: {duplicate_dir}")
                print(f"   正确目录: {correct_dir}")

                moved_count = 0
                for file_name in os.listdir(duplicate_dir):
                    src_file = os.path.join(duplicate_dir, file_name)
                    dst_file = os.path.join(correct_dir, file_name)

                    try:
                        if os.path.isfile(src_file):
                            # 如果目标文件已存在，重命名
                            if os.path.exists(dst_file):
                                base, ext = os.path.splitext(file_name)
                                new_name = f"{base}_dup{ext}"
                                dst_file = os.path.join(correct_dir, new_name)

                            shutil.move(src_file, dst_file)
                            moved_count += 1
                            print(f"   移动文件: {file_name}")
                    except Exception as e:
                        print(f"   移动文件 {file_name} 失败: {e}")

                if moved_count > 0:
                    print(f"✅ 从重复目录移动了 {moved_count} 个文件到正确位置")

                # 尝试删除空目录
                try:
                    if not os.listdir(duplicate_dir):
                        os.rmdir(duplicate_dir)
                        print(f"🗑️ 删除空目录: {duplicate_dir}")
                except:
                    pass

    def _cleanup_old_charts(self):
        """清理过期的图表文件"""
        if not self.config.CHART_CACHE_ENABLED:
            return

        charts_dir = self.config.CHARTS_DIR
        if not os.path.exists(charts_dir):
            return

        import time
        current_time = time.time()
        expiry_seconds = self.config.CHART_CACHE_EXPIRY_HOURS * 3600

        files_removed = 0
        for file_name in os.listdir(charts_dir):
            file_path = os.path.join(charts_dir, file_name)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > expiry_seconds:
                    try:
                        os.remove(file_path)
                        files_removed += 1
                        print(f"清理过期图表: {file_name}")
                    except:
                        pass

        if files_removed > 0:
            print(f"🗑️ 清理了 {files_removed} 个过期图表文件")

    def process(self, input_messages: List[Dict[str, Any]], tasks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理任务"""
        print(f"\n{'=' * 60}")
        print("🚀 Worker Agent 开始处理任务")
        print(f"{'=' * 60}")

        if tasks:
            print(f"📋 接收 {len(tasks)} 个任务:")
            for i, task in enumerate(tasks):
                print(f"  {i + 1}. {task.get('task_name')} ({task.get('task_id')})")

        try:
            messages = []
            for msg in input_messages:
                if isinstance(msg, dict):
                    messages.append(AIMessage(content=json.dumps(msg)))
                elif isinstance(msg, AIMessage):
                    messages.append(msg)

            state: State = {
                "messages": messages,
                "tasks": tasks or [],
                "current_task": None,
                "current_result": None,
                "all_results": [],
                "cursor": 0,
                "worker_state": {"next_node": "router_node"},
                "tool_execution_history": [],
                "available_tools": [],
                "knowledge_base_initialized": False,
                "spider_initialized": False
            }

            # 使用循环模拟子图的执行，实现 cursor 自增逻辑
            # 这确保了 WorkerAgent 在作为独立工具调用时能够完成所有任务
            while True:
                state = self.workflow.invoke(state, {"recursion_limit": 100})
                
                # 检查最新产出的消息类型
                last_result = self._extract_result(state)
                msg_type = last_result.get("message", {}).get("type")
                
                if msg_type == "TASK_RESULT":
                    # 任务完成，游标自增，准备执行下一个或生成最终结果
                    state["cursor"] += 1
                    state["worker_state"]["next_node"] = "router_node"
                    print(f"✅ 任务 {state['cursor']} 完成，游标移至 {state['cursor']}")
                elif msg_type == "FINAL_RESULT":
                    # 最终报告生成完成，退出循环
                    break
                elif state.get("worker_state", {}).get("next_node") == "end":
                    # 异常情况或无更多任务
                    break

            result = self._extract_result(state)

            print(f"\n✅ Worker Agent 任务处理完成")
            print(f"   最终状态: {result.get('status')}")
            print(f"   知识库使用: {result.get('knowledge_base_used', False)}")
            print(f"   爬虫使用: {result.get('spider_results_used', False)}")
            print(f"   图表保存目录: {self.config.CHARTS_DIR}")

            return result

        except Exception as e:
            print(f"❌ 任务处理失败: {str(e)}")
            import traceback
            traceback.print_exc()

            return {
                "status": "ERROR",
                "error": str(e),
                "message": None,
                "all_results": []
            }

    def _extract_result(self, state: State) -> Dict[str, Any]:
        """从状态提取结果"""
        last_msg_type = None
        output_msg = None

        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                try:
                    data = json.loads(msg.content)
                    msg_type = data.get("type")
                    if msg_type in ["TASK_RESULT", "FINAL_RESULT"]:
                        last_msg_type = msg_type
                        output_msg = data
                        break
                except:
                    continue

        if last_msg_type == "FINAL_RESULT":
            status = "COMPLETED"
        elif last_msg_type == "TASK_RESULT":
            status = "IN_PROGRESS"
        else:
            status = "UNKNOWN"

        worker_state = state.get("worker_state", {})

        return {
            "status": status,
            "message": output_msg,
            "all_results": state.get("all_results", []),
            "current_result": state.get("current_result"),
            "progress": worker_state.get("progress", 0.0),
            "current_section": worker_state.get("current_section", ""),
            "tool_usage_stats": worker_state.get("tool_usage_stats", {}),
            "knowledge_base_used": worker_state.get("knowledge_base_used", False),
            "spider_results_used": worker_state.get("spider_results_used", False)
        }


# ==================== 9. 使用示例 ====================

if __name__ == "__main__":
    print("初始化Worker Agent...")
    agent = WorkerAgent()

    print("=" * 60)
    print("✅ Worker Agent 初始化完成")
    print(f"   报告目录: {agent.config.REPORT_DIR}")
    print(f"   图表目录: {agent.config.CHARTS_DIR}")
    print("=" * 60)

    test_tasks = [
        {
            "task_id": "task_001_dataset_overview",
            "task_name": "电力数据集概览与基本统计对比",
            "task_type": "comparative_analysis",
            "task_description": """作为电力系统数据分析专家，请对ETTh2和ETTm1两个数据集进行全面的基础统计对比分析。

任务要求：
1. 数据集基本信息对比
   - 分别展示两个数据集的数据规模、时间范围、采样频率
   - 对比数据完整性（缺失值情况）

2. 电力参数统计特征对比
   - 电压（HUFL、HULL）的平均值、标准差、极值对比
   - 电流（MUFL、MULL）的分布特征对比
   - 负荷（LUFL、LULL）的基本统计对比

3. 数据质量评估
   - 评估两个数据集的数据质量
   - 识别异常值和缺失数据模式
   - 提出数据预处理建议

可视化要求：
- 生成两个数据集的总体统计对比图（箱线图或雷达图）
- 生成主要电力参数的分布直方图对比
- 生成数据完整性热力图对比
- 生成统计特征对比表格

报告要求：
- 800-1000字的详细对比分析报告
- 使用专业的电力系统术语
- 突出两个数据集的差异特征
- 基于统计特征提出分析建议""",
            "use_rag": True,
            "generate_table": True,
            "generate_figure": True,
            "use_resources": ["data/ETTh2.csv", "data/ETTm1.csv"],
            "query": "电力数据集统计特征对比分析"
        }
    ]

    print(f"\n测试任务: {test_tasks[0]['task_name']}")
    print(f"任务描述: {test_tasks[0]['task_description'][:100]}...")
    print(f"任务ID: {test_tasks[0]['task_id']}")

    print("\n" + "=" * 60)
    print("🚀 Agent 准备就绪，开始执行任务")
    print("=" * 60)

    test_input_messages = [{
        "type": "PLAN_RESULT",
        "content": {
            "tasks": test_tasks,
            "cursor": 0
        }
    }]

    result = agent.process(test_input_messages, test_tasks)

    print("\n" + "=" * 60)
    print("📊 测试结果摘要")
    print("=" * 60)
    print(f"状态: {result.get('status')}")
    print(f"进度: {result.get('progress', 0) * 100:.1f}%")
    print(f"结果数量: {len(result.get('all_results', []))}")
    print(f"知识库使用: {result.get('knowledge_base_used', False)}")
    print(f"爬虫使用: {result.get('spider_results_used', False)}")

    if result.get('message') and result['message'].get('content'):
        report_paths = result['message']['content'].get('report_paths', {})
        if report_paths:
            print(f"\n📄 生成的报告文件:")
            for report_name, report_path in report_paths.items():
                print(f"  {report_name}: {report_path}")

    charts_dir = agent.config.CHARTS_DIR
    if os.path.exists(charts_dir):
        chart_files = [f for f in os.listdir(charts_dir) if f.endswith(('.png', '.svg'))]
        if chart_files:
            print(f"\n📈 生成的图表文件 ({len(chart_files)} 个):")
            for chart_file in chart_files[:5]:
                print(f"  📊 {chart_file}")
            if len(chart_files) > 5:
                print(f"  还有 {len(chart_files) - 5} 个图表文件")
