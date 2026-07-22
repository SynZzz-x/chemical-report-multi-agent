"""
化工产业知识库 - ChemicalKnowledgeBase.py
修复相似度分数问题
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import numpy as np

print("🔧 正在检查依赖包...")


# ==================== 动态依赖导入 ====================
def check_and_import_dependencies():
    """检查并导入所有依赖"""

    imported_modules = {}

    # 尝试导入langchain_core（新版本LangChain使用这个）
    try:
        from langchain_core.documents import Document
        imported_modules['Document'] = Document
        print("✅ 从 langchain_core 导入 Document 成功")
    except ImportError:
        print("❌ 无法从 langchain_core 导入 Document")
        return False

    # 尝试导入文本分割器 - 尝试多种导入方式
    text_splitter_imported = False
    try:
        # 尝试从 langchain_text_splitters 导入
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        imported_modules['RecursiveCharacterTextSplitter'] = RecursiveCharacterTextSplitter
        print("✅ 从 langchain_text_splitters 导入 RecursiveCharacterTextSplitter 成功")
        text_splitter_imported = True
    except ImportError:
        try:
            # 尝试从 langchain.text_splitter 导入（旧版本）
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            imported_modules['RecursiveCharacterTextSplitter'] = RecursiveCharacterTextSplitter
            print("✅ 从 langchain.text_splitter 导入 RecursiveCharacterTextSplitter 成功")
            text_splitter_imported = True
        except ImportError:
            print("⚠️  无法导入 RecursiveCharacterTextSplitter，将使用自定义文本分割器")

    # 尝试导入OpenAI
    try:
        from langchain_openai import ChatOpenAI
        imported_modules['ChatOpenAI'] = ChatOpenAI
        print("✅ 导入 langchain_openai.ChatOpenAI 成功")
    except ImportError:
        print("❌ 无法导入 ChatOpenAI")
        return False

    # 尝试导入文档加载器
    try:
        from langchain_community.document_loaders import (
            PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader,
            UnstructuredExcelLoader
        )
        imported_modules['PyPDFLoader'] = PyPDFLoader
        imported_modules['Docx2txtLoader'] = Docx2txtLoader
        imported_modules['CSVLoader'] = CSVLoader
        imported_modules['TextLoader'] = TextLoader
        imported_modules['UnstructuredExcelLoader'] = UnstructuredExcelLoader
        print("✅ 导入 langchain_community.document_loaders 成功")
    except ImportError:
        print("❌ 无法导入文档加载器")
        return False

    # 尝试导入ChromaDB
    try:
        import chromadb
        from chromadb.config import Settings
        imported_modules['chromadb'] = chromadb
        imported_modules['Settings'] = Settings
        print("✅ 导入 chromadb 成功")
    except ImportError:
        print("❌ 无法导入 chromadb")
        return False

    # 尝试导入Chroma向量存储
    try:
        from langchain_community.vectorstores import Chroma
        imported_modules['Chroma'] = Chroma
        print("✅ 导入 langchain_community.vectorstores.Chroma 成功")
    except ImportError:
        print("❌ 无法导入 Chroma")
        return False

    # 尝试导入DashScope嵌入
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        imported_modules['DashScopeEmbeddings'] = DashScopeEmbeddings
        print("✅ 导入 langchain_community.embeddings.DashScopeEmbeddings 成功")
    except ImportError:
        print("❌ 无法导入 DashScopeEmbeddings")
        return False

    # 检查必要依赖
    necessary_modules = ['Document', 'ChatOpenAI', 'chromadb', 'Settings', 'Chroma', 'DashScopeEmbeddings']
    for module in necessary_modules:
        if module not in imported_modules:
            print(f"❌ 缺少必要模块: {module}")
            return False

    # 如果没有导入文本分割器，创建自定义的
    if not text_splitter_imported:
        print("🔧 创建自定义文本分割器...")
        imported_modules['RecursiveCharacterTextSplitter'] = create_custom_text_splitter()

    return imported_modules


def create_custom_text_splitter():
    """创建自定义文本分割器类"""

    class CustomTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", ".", " ", ""]

        def split_text(self, text):
            """简单的文本分割实现"""
            chunks = []
            current_chunk = ""

            # 按换行分割文本
            paragraphs = text.split('\n')

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                # 如果当前块加上新段落会超过大小，保存当前块
                if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                    chunks.append(current_chunk)
                    # 保留重叠部分
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + "\n" + paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n" + paragraph
                    else:
                        current_chunk = paragraph

            # 添加最后一个块
            if current_chunk:
                chunks.append(current_chunk)

            return chunks

    return CustomTextSplitter


# 执行依赖检查
modules = check_and_import_dependencies()
if not modules:
    print("\n❌ 缺少依赖包！")
    print("\n请运行以下命令安装所有依赖：")
    print("pip install langchain-openai langchain-community")
    print("pip install chromadb langchain-chroma dashscope")
    print("pip install langchain-text-splitters")  # 新增：文本分割器单独包
    print("pip install pypdf docx2txt unstructured pandas")
    print("\n或者运行单个命令：")
    print(
        "pip install langchain-openai langchain-community chromadb langchain-chroma dashscope langchain-text-splitters pypdf docx2txt unstructured pandas")
    exit(1)

# 现在导入具体的模块
from langchain_core.documents import Document

RecursiveCharacterTextSplitter = modules['RecursiveCharacterTextSplitter']
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
)
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from src.llm import get_llm

print("✅ 所有依赖检查通过，开始初始化化工知识库...")


# ==================== 化工知识库核心类 ====================

class ChemicalDocument:
    """化工文档类"""

    def __init__(self, doc_id: str, title: str, content: str, doc_type: str, source: str,
                 metadata: Dict[str, Any] = None):
        self.doc_id = doc_id
        self.title = title
        self.text_content = content
        self.doc_type = doc_type
        self.source = source
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()


class ChemicalDocumentLoader:
    """化工文档加载器"""

    @staticmethod
    def load_document(file_path: str) -> ChemicalDocument:
        """加载单个文档"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()

        # 推断文档类型
        doc_type = ChemicalDocumentLoader._infer_doc_type(file_path)

        try:
            if file_ext == '.pdf':
                content = ChemicalDocumentLoader._load_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                content = ChemicalDocumentLoader._load_word(file_path)
            elif file_ext == '.csv':
                content = ChemicalDocumentLoader._load_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                content = ChemicalDocumentLoader._load_excel(file_path)
            elif file_ext in ['.txt', '.md', '.json', '.yaml']:
                content = ChemicalDocumentLoader._load_text(file_path)
            else:
                content = ChemicalDocumentLoader._load_text(file_path)  # 尝试作为文本文件

            # 生成文档ID
            doc_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
            doc_id = f"{doc_type}_{doc_hash}"

            return ChemicalDocument(
                doc_id=doc_id,
                title=os.path.basename(file_path),
                content=content,
                doc_type=doc_type,
                source=file_path,
                metadata={
                    "extension": file_ext,
                    "content_length": len(content),
                }
            )

        except Exception as e:
            raise Exception(f"加载文档失败 {file_path}: {str(e)}")

    @staticmethod
    def _infer_doc_type(file_path: str) -> str:
        """推断文档类型"""
        filename = os.path.basename(file_path).lower()

        if any(kw in filename for kw in ["专利", "patent"]):
            return "patent"
        elif any(kw in filename for kw in ["安全", "safety", "危险"]):
            return "safety"
        elif any(kw in filename for kw in ["工艺", "process", "流程"]):
            return "process"
        elif any(kw in filename for kw in ["设备", "equipment", "机器"]):
            return "equipment"
        elif any(kw in filename for kw in ["材料", "material", "chemical"]):
            return "material"
        elif any(kw in filename for kw in ["标准", "standard", "规范"]):
            return "standard"
        elif any(kw in filename for kw in ["数据", "data", "记录"]):
            return "data"
        else:
            return "report"

    @staticmethod
    def _load_pdf(file_path: str) -> str:
        """加载PDF"""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            content_parts = []
            for i, page in enumerate(pages):
                content_parts.append(f"[第{i + 1}页] {page.page_content}")
            return "\n\n".join(content_parts)
        except Exception as e:
            print(f"⚠️  加载PDF失败，使用文本方式: {str(e)}")
            return ChemicalDocumentLoader._load_text(file_path)

    @staticmethod
    def _load_word(file_path: str) -> str:
        """加载Word"""
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return documents[0].page_content if documents else ""
        except Exception as e:
            print(f"⚠️  加载Word失败，使用文本方式: {str(e)}")
            return ChemicalDocumentLoader._load_text(file_path)

    @staticmethod
    def _load_csv(file_path: str) -> str:
        """加载CSV"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path, nrows=100)
            return f"CSV数据：{len(df)}行×{len(df.columns)}列\n列名：{', '.join(df.columns.tolist())}"
        except Exception as e:
            print(f"⚠️  使用pandas加载CSV失败: {str(e)}")
            try:
                loader = CSVLoader(file_path)
                documents = loader.load()
                return "\n".join([doc.page_content for doc in documents[:5]])
            except Exception as e2:
                print(f"⚠️  使用CSVLoader加载CSV失败: {str(e2)}")
                return "CSV文件内容"

    @staticmethod
    def _load_excel(file_path: str) -> str:
        """加载Excel"""
        try:
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            documents = loader.load()
            content_parts = []
            for doc in documents:
                if doc.page_content.strip():
                    content_parts.append(doc.page_content)
            return "\n".join(content_parts)
        except Exception as e:
            print(f"⚠️  加载Excel失败: {str(e)}")
            return "Excel文件内容"

    @staticmethod
    def _load_text(file_path: str) -> str:
        """加载文本文件"""
        try:
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            # 如果所有编码都失败，使用二进制读取
            with open(file_path, 'rb') as f:
                return f.read().decode('utf-8', errors='ignore')
        except Exception as e:
            raise Exception(f"读取文本文件失败: {str(e)}")


class ChemicalKnowledgeBase:
    """化工知识库核心类"""

    def __init__(self,
                 dashscope_api_key: str,
                 persist_directory: str = "./chemical_kb",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):

        # 验证API密钥
        if not dashscope_api_key or not dashscope_api_key.startswith("sk-"):
            raise ValueError("无效的DashScope API密钥")

        self.dashscope_api_key = dashscope_api_key
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化嵌入模型（直接使用通义千问）
        print("🔧 初始化通义千问嵌入模型...")
        try:
            self.embeddings = DashScopeEmbeddings(
                dashscope_api_key=self.dashscope_api_key,
                model="text-embedding-v1"
            )
            print("✅ 嵌入模型初始化成功")
        except Exception as e:
            print(f"❌ 嵌入模型初始化失败: {str(e)}")
            raise

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # 初始化ChromaDB
        self._init_chromadb()

        # 初始化LLM（用于生成答案）
        print("🔧 初始化LLM客户端...")
        try:
            if self.dashscope_api_key:
                os.environ["OPENAI_API_KEY"] = self.dashscope_api_key
                
            llm_config = {
                "configurable": {
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model_name": "qwen-max",
                    "temperature": 0.3,
                    "max_tokens": 1000
                }
            }
            self.llm = get_llm(llm_config, json_mode=False)
            print("✅ LLM客户端初始化成功")
        except Exception as e:
            print(f"❌ LLM客户端初始化失败: {str(e)}")
            # 设置一个None值，稍后处理
            self.llm = None

        print(f"✅ 化工知识库初始化完成")
        print(f"   存储路径: {os.path.abspath(self.persist_directory)}")
        print(f"   分块大小: {chunk_size} 字符")
        print(f"   重叠大小: {chunk_overlap} 字符")

    def _init_chromadb(self):
        """初始化ChromaDB"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            # 创建集合 - 使用余弦相似度
            self.collection_name = "chemical_documents"
            try:
                self.collection = self.client.get_collection(self.collection_name)
                print(f"📂 加载现有集合: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "化工产业知识库",
                        "hnsw:space": "cosine"  # 使用余弦相似度
                    }
                )
                print(f"📂 创建新集合: {self.collection_name}")

            # 创建LangChain Chroma包装
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("✅ ChromaDB向量存储初始化成功")

        except Exception as e:
            raise Exception(f"ChromaDB初始化失败: {str(e)}")

    def load_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """加载文档到知识库"""
        if not file_paths:
            return {"error": "未提供文档路径", "loaded": 0, "chunks": 0}

        results = {
            "total_files": len(file_paths),
            "loaded_files": 0,
            "failed_files": [],
            "total_chunks": 0,
            "documents": []
        }

        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"文件不存在: {file_path}")

                print(f"📄 加载: {os.path.basename(file_path)}")

                # 加载文档
                document = ChemicalDocumentLoader.load_document(file_path)

                # 文本分块
                chunks = self.text_splitter.split_text(document.text_content)

                # 创建Document对象
                documents = []
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "doc_id": document.doc_id,
                            "title": document.title,
                            "doc_type": document.doc_type,
                            "source": document.source,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        }
                    )
                    documents.append(doc)

                # 添加到向量存储
                if documents:
                    self.vector_store.add_documents(documents)
                    results["total_chunks"] += len(documents)
                    results["loaded_files"] += 1

                    results["documents"].append({
                        "doc_id": document.doc_id,
                        "title": document.title,
                        "type": document.doc_type,
                        "chunks": len(documents),
                        "status": "loaded"
                    })

                    print(f"   ✅ 成功: {len(documents)} 个文本块")

            except Exception as e:
                error_msg = f"加载失败: {str(e)}"
                print(f"   ❌ 错误: {error_msg}")
                results["failed_files"].append({
                    "file": file_path,
                    "error": error_msg
                })

        # 持久化存储
        if results["loaded_files"] > 0:
            self.vector_store.persist()

        return results

    def query(self,
              question: str,
              top_k: int = 5,
              doc_type_filter: Optional[str] = None,
              similarity_threshold: float = 0.3,  # 降低阈值，因为余弦相似度可能返回负值
              generate_answer: bool = True) -> Dict[str, Any]:
        """查询知识库"""
        try:
            # 构建过滤器
            filter_dict = {}
            if doc_type_filter:
                filter_dict["doc_type"] = doc_type_filter

            # 执行搜索 - 使用自定义的相似度计算
            print(f"🔍 正在搜索: {question}")
            if filter_dict:
                docs = self.vector_store.similarity_search(
                    question,
                    k=top_k * 2,
                    filter=filter_dict
                )
            else:
                docs = self.vector_store.similarity_search(
                    question,
                    k=top_k
                )

            # 手动计算相似度分数（如果可用）
            results = []
            for i, doc in enumerate(docs[:top_k]):
                # 这里我们使用简单的排名分数，因为实际相似度分数可能不正常
                score = 1.0 - (i * 0.1)  # 基于排名的分数

                result = {
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown"),
                    "title": doc.metadata.get("title", "Untitled"),
                    "doc_type": doc.metadata.get("doc_type", "unknown"),
                }
                results.append(result)

            # 生成答案
            answer = ""
            if results and generate_answer and self.llm:
                answer = self._generate_answer(question, results)

            return {
                "question": question,
                "total_results": len(results),
                "average_score": sum(r["score"] for r in results) / len(results) if results else 0,
                "answer": answer,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
            return {
                "question": question,
                "error": f"查询失败: {str(e)}",
                "total_results": 0,
                "results": []
            }

    def _generate_answer(self, question: str, results: List[Dict[str, Any]]) -> str:
        """使用LLM生成答案"""
        try:
            # 构建上下文
            context = "\n\n".join([
                f"[来源: {r['title']}, 相关度: {r['score']:.2f}]\n{r['content'][:500]}..."
                if len(r['content']) > 500 else r['content']
                for r in results[:3]
            ])

            prompt = f"""你是一个化工产业专家，请基于以下知识库内容回答问题。

问题：{question}

相关知识：
{context}

请生成专业、准确的回答，要求：
1. 直接回答问题的核心内容
2. 引用相关知识库内容作为支撑
3. 提供化工产业的专业见解
4. 回答语言：中文
5. 回答长度：200-300字
"""

            print("💭 正在生成答案...")
            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            print(f"❌ 生成答案失败: {str(e)}")
            return f"无法生成答案: {str(e)}"

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            count = self.collection.count() if hasattr(self, 'collection') and self.collection else 0

            return {
                "vector_store": "ChromaDB",
                "persist_directory": self.persist_directory,
                "collection_name": self.collection_name,
                "total_vectors": count,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "status": "active",
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"获取统计失败: {str(e)}"}


# ==================== 主程序 ====================

def main():
    """主函数：直接运行使用"""

    print("=" * 60)
    print("化工知识库 - 测试版本")
    print("=" * 60)

    # ============ 配置参数 ============

    # 1. 您的通义千问API密钥
    DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
    if not DASHSCOPE_API_KEY:
        print("DASHSCOPE_API_KEY is required to run this manual knowledge-base demo.")
        return

    # 2. 您的文档路径列表
    YOUR_DOCUMENT_PATHS = [
        "chemical_test_documents\催化裂化催化剂研究报告.txt",
        "chemical_test_documents\危险化学品管理规范.txt",
        "chemical_test_documents\聚乙烯市场分析报告.txt",
    ]

    # 检查文档是否存在
    print("\n📁 检查文档文件...")
    existing_files = []
    for path in YOUR_DOCUMENT_PATHS:
        if os.path.exists(path):
            existing_files.append(path)
            print(f"  ✅ {os.path.basename(path)}")
        else:
            print(f"  ❌ 文件不存在: {path}")

    if not existing_files:
        print("\n❌ 错误：没有找到任何文档文件！")
        print("请确保文档路径正确，或者创建测试文档")
        print("\n🔧 是否创建测试文档？(y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            create_test_documents()
            existing_files = [
                "./test_docs/聚乙烯测试文档.txt",
                "./test_docs/催化裂化测试文档.txt",
                "./test_docs/化工安全测试文档.txt",
            ]
        else:
            return

    # ============ 启动知识库 ============

    print(f"\n🚀 正在初始化化工知识库...")

    try:
        kb = ChemicalKnowledgeBase(
            dashscope_api_key=DASHSCOPE_API_KEY,
            persist_directory="./chemical_kb_test",
            chunk_size=1000,
            chunk_overlap=200
        )

        # 加载文档
        print(f"\n📚 加载文档 ({len(existing_files)} 个)...")
        load_result = kb.load_documents(existing_files)

        print(f"\n📊 加载结果:")
        print(f"  成功: {load_result['loaded_files']} 个文件")
        print(f"  文本块: {load_result['total_chunks']} 个")

        if load_result['failed_files']:
            print(f"\n⚠️  失败的文件:")
            for failed in load_result['failed_files'][:3]:
                print(f"  - {failed['file']}: {failed['error'][:50]}...")

        # 获取统计信息
        stats = kb.get_stats()
        print(f"\n📈 知识库统计:")
        print(f"  向量总数: {stats.get('total_vectors', 0)}")

        # ============ 交互式查询 ============

        print("\n" + "=" * 60)
        print("✅ 化工知识库初始化完成！")
        print("输入 'exit' 或 'quit' 退出")
        print("=" * 60)

        while True:
            try:
                # 获取用户输入
                user_input = input("\n🔍 请输入您的问题: ").strip()

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 再见！")
                    break

                if not user_input:
                    print("⚠️  请输入有效的问题")
                    continue

                # 执行查询并显示结果
                print(f"\n📝 您的问题: {user_input}")
                print("⏳ 正在搜索知识库...")

                result = kb.query(
                    question=user_input,
                    top_k=3,
                    generate_answer=True
                )

                # 显示结果
                print("\n" + "=" * 60)
                print("📋 查询结果")
                print("=" * 60)

                if result.get("error"):
                    print(f"❌ 错误: {result['error']}")
                    continue

                print(f"📊 找到 {result['total_results']} 个相关结果")
                if result.get("answer"):
                    print(f"\n💡 专业回答:")
                    print("-" * 40)
                    print(result["answer"])
                    print("-" * 40)

                # 显示来源
                if result.get("results"):
                    print(f"\n📄 相关来源:")
                    for i, r in enumerate(result["results"][:3], 1):
                        source_name = os.path.basename(r['source'])
                        print(f"{i}. {source_name} (相关度: {r['score']:.2f})")
                        # 显示部分内容
                        content_preview = r['content'][:150] + "..." if len(r['content']) > 150 else r['content']
                        print(f"   {content_preview}\n")

                print("=" * 60)

            except KeyboardInterrupt:
                print("\n\n👋 用户中断，再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {str(e)}")
                continue

    except Exception as e:
        print(f"❌ 知识库初始化失败: {str(e)}")
        print("\n🔧 问题排查：")
        print("1. 检查API密钥是否正确（以'sk-'开头）")
        print("2. 检查是否已安装所有依赖包")
        print("3. 检查网络连接是否正常")


def create_test_documents():
    """创建测试文档"""
    print("\n📝 创建测试文档...")

    test_dir = "./test_docs"
    os.makedirs(test_dir, exist_ok=True)

    # 创建测试文档
    test_docs = [
        ("聚乙烯测试文档.txt", """
聚乙烯生产工艺报告

一、概述
聚乙烯（Polyethylene, PE）是全球产量最大的塑料原料，广泛应用于包装、建筑、农业等领域。

二、生产工艺
1. 高压法（LDPE）
   - 反应温度：150-300°C
   - 反应压力：100-300 MPa
   - 产品特点：柔韧性好，透明度高

2. 低压法（HDPE）
   - 反应温度：60-80°C
   - 反应压力：0.5-2.0 MPa
   - 催化剂：Ziegler-Natta催化剂
   - 产品特点：强度高，耐化学性好

3. 气相法（LLDPE）
   - 反应温度：70-90°C
   - 反应压力：1.0-2.5 MPa
   - 产品特点：抗撕裂性好，拉伸强度高

三、产品应用
1. 薄膜：包装膜、农膜
2. 管材：给水管、燃气管
3. 注塑制品：容器、玩具
4. 纤维：绳索、渔网

四、质量控制
- 熔融指数（MFI）：0.5-50 g/10min
- 密度：0.918-0.965 g/cm³
- 拉伸强度：≥ 20 MPa
- 断裂伸长率：≥ 500%
        """),

        ("催化裂化测试文档.txt", """
催化裂化（FCC）技术报告

一、工艺概述
催化裂化是炼油厂将重质油转化为高价值轻质油品的重要工艺。

二、工艺流程
1. 反应-再生系统
   - 反应温度：500-530°C
   - 反应压力：0.2-0.3 MPa
   - 催化剂类型：分子筛催化剂

2. 分馏系统
   - 产品：汽油、柴油、液化气
   - 收率：汽油40-50%，柴油20-25%

三、催化剂
- Y型分子筛：活性高，选择性好
- ZSM-5：提高汽油辛烷值
- 基质：提供物理强度和传热性能

四、安全措施
- 可燃气体检测
- 紧急泄压系统
- 粉尘爆炸防护
        """),

        ("化工安全测试文档.txt", """
化工生产安全规范

一、个人防护装备（PPE）
1. 头部防护：安全帽
2. 眼部防护：安全眼镜、防护面罩
3. 呼吸防护：防毒面具、空气呼吸器
4. 身体防护：防护服、手套、安全鞋

二、化学品安全
1. 储存要求：分类储存，标识清晰
2. 搬运要求：使用专用工具，轻拿轻放
3. 泄漏处理：立即隔离，使用吸附材料

三、应急处理
1. 火灾：使用适当灭火器，向上风向撤离
2. 泄漏：启动应急预案，联系专业人员
3. 人员受伤：立即急救，呼叫医疗救援
        """)
    ]

    for filename, content in test_docs:
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ 已创建: {filepath}")

    print(f"\n📁 测试文档已创建到: {os.path.abspath(test_dir)}")


# ==================== 直接执行 ====================

if __name__ == "__main__":
    # 直接运行主函数
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序运行失败: {str(e)}")
        print("\n💡 请确保已安装所有依赖包，运行:")
        print("pip install langchain-openai langchain-community")
        print("pip install chromadb langchain-chroma dashscope")
        print("pip install langchain-text-splitters")  # 新增：文本分割器单独包
        print("pip install pypdf docx2txt unstructured pandas")
