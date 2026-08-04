import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from ..state import State
from ..llm import get_llm
from ..utils import md_to_docx, md_to_pdf, md_rewrite
from ..utils.path_manager import get_session_cache_dir
from ..evidence.reporting import append_missing_figures, format_evidence_table

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Summarizer V2 Node:
- Refactored to use LLM for generating markdown sections based on tasks and results.
- Embeds charts and tables using <description> tags.
- Generates a final report in Markdown, PDF, and DOCX formats.
"""

def _read_prompt(rel_path: str) -> str:
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, rel_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read prompt from {path}: {e}")
        # Return a fallback or empty string, though this should ideally fail hard if prompt is missing
        return ""

def find_title(state: State) -> str:
    for msg in state.get("messages", []) or []:
        try:
            content = json.loads(msg.content)
        except Exception:
            continue
        if content.get("from") == "Intake" and content.get("to") == "Planner":
            return content.get("title") or "自动生成报告"
    return "自动生成报告"

def _content_reorganizer(state: State) -> List[Dict[str, Any]]:
    sections = []
    tasks = state.get("tasks", []) or []
    for res in state.get("results", []) or []:
        # 任务名作为章节标题（若找不到则使用 task_id）
        title = None
        for t in tasks:
            if t.get("task_id") == res.get("task_id"):
                title = t.get("task_name")
                break
        title = title or res.get("task_id") or "章节"

        # 优先使用 res 中的 figures (包含详细描述)
        figures = res.get("figures", [])
        if not figures:
            # Fallback: 从 outputs 提取
            for p in (res.get("outputs", []) or []):
                if str(p).lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
                    figures.append({"path": p, "description": f"图像：{os.path.basename(p)}"})
        for figure in figures:
            if figure.get("graph_type") and figure.get("evidence_ids"):
                markers = "、".join(f"[{value}]" for value in figure["evidence_ids"])
                description = str(figure.get("description") or "概念关系图")
                if markers not in description:
                    figure["description"] = f"{description}（关系证据：{markers}）"

        section = {
            "title": title,
            "text": res.get("text_output", ""),
            "tables": res.get("tables", []),
            "figures": figures,
            "citations": res.get("citations", []),
            "notes": res.get("notes", ""),
        }
        sections.append(section)
    return sections

def _downgrade_headings(content: str) -> str:
    """
    Downgrade all markdown headings by one level.
    # Title -> ## Title
    ## Title -> ### Title
    """
    lines = content.split('\n')
    new_lines = []
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
            
        if in_code_block:
            new_lines.append(line)
            continue
            
        # Check for headers: starts with # followed by space
        match = re.match(r'^(\s*)(#+)(\s.*)', line)
        if match:
            indent = match.group(1)
            hashes = match.group(2)
            rest = match.group(3)
            new_lines.append(f"{indent}#{hashes}{rest}")
        else:
            new_lines.append(line)
            
    return "\n".join(new_lines)

def _generate_section_content(section: Dict[str, Any], config: RunnableConfig) -> str:
    """
    Use LLM to generate the markdown content for a single section.
    """
    model = get_llm(config)
    prompt_template = _read_prompt("../prompts/summarizer_section_writer.md")
    
    if not prompt_template:
        logger.warning("Prompt template not found. Using raw text.")
        raw = f"## {section['title']}\n\n{section['text']}"
        raw = append_missing_figures(raw, section.get("figures", []))
        evidence_table = format_evidence_table(section.get("citations", []))
        return f"{raw}\n\n{evidence_table}" if evidence_table else raw

    # Prepare data for the prompt
    section_title = section.get("title", "Section")
    text_content = section.get("text", "")
    
    # Serialize figures and tables to JSON for the prompt
    figures_json = json.dumps(section.get("figures", []), ensure_ascii=False, indent=2)
    tables_json = json.dumps(section.get("tables", []), ensure_ascii=False, indent=2)
    citations = section.get("citations", [])
    citations_json = json.dumps(citations, ensure_ascii=False, indent=2)

    def with_evidence(value: str) -> str:
        value = append_missing_figures(value, section.get("figures", []))
        table = format_evidence_table(citations)
        return f"{value.rstrip()}\n\n{table}" if table else value
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", "请根据以下信息对章节内容进行排版和图表嵌入（保持原文）：\n\n章节标题：{section_title}\n\n文本素材：\n{text_content}\n\n可用图片资源：\n{figures_list}\n\n可用表格资源：\n{tables_list}\n\n引用证据（不得删除或改写证据编号）：\n{citations_list}")
    ])
    
    try:
        messages = prompt.format_messages(
            section_title=section_title,
            text_content=text_content,
            figures_list=figures_json,
            tables_list=tables_json,
            citations_list=citations_json,
        )
        resp = model.invoke(messages, config=config)
        content = str(getattr(resp, "content", "")).strip()
        
        # Remove potential markdown code block wrappers
        if content.startswith("```markdown") and content.endswith("```"):
            content = content[11:-3]
        elif content.startswith("```") and content.endswith("```"):
            # 检查第一行是否仅为 ``` (无语言标识)
            # 只有当第一行纯粹是 ``` 时才认为是 wrapper，防止误删 ```python ... ```
            lines = content.split('\n', 1)
            if lines and lines[0].strip() == "```":
                content = content[3:-3]
            
        content = content.strip()
        
        # 检查 LLM 输出是否已经包含标题 (H1 或 H2)
        if content.startswith("# "):
            # 如果内容以一级标题开头，说明 LLM 生成了错误的层级结构
            # 此时需要将全文所有标题降级 (H1->H2, H2->H3...)
            return with_evidence(_downgrade_headings(content))
            
        elif content.startswith("## "):
            # 已有 H2，直接返回
            return with_evidence(content)
            
        else:
            # 没有标题，手动添加
            return with_evidence(f"## {section_title}\n\n{content}")
    except Exception as e:
        logger.error(f"LLM generation failed for section {section_title}: {e}")
        # Fallback: 如果原始内容以 H1 开头，也进行降级处理
        stripped_text = text_content.strip()
        if stripped_text.startswith("# "):
            return with_evidence(_downgrade_headings(text_content) + "\n\n(LLM Generation Failed)")
        return with_evidence(f"## {section_title}\n\n{text_content}\n\n(LLM Generation Failed)")

def _generate_report_evaluation(report_text: str, config: RunnableConfig) -> str:
    """
    Generate a brief evaluation of the report using LLM.
    """
    try:
        model = get_llm(config)
        sys_prompt = _read_prompt("../prompts/summarizer_eval.md")
        if not sys_prompt:
             sys_prompt = "你是一位专业的报告评价专家。阅读报告全文，按结构/完整性/专业性/总体评价四个维度，用约200字输出自然语言评价。"
             
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "{report}")
        ])
        
        # Truncate report if too long to avoid token limits (simple truncation)
        # Assuming 1 char ~= 1 token roughly for safety, keep last 10000 chars or first+last?
        # Usually summary needs full context, but if too large, we might just take the first 20000 chars.
        truncated_report = report_text[:30000] 
        
        messages = prompt.format_messages(report=truncated_report)
        resp = model.invoke(messages, config=config)
        return str(getattr(resp, "content", "")).strip()
    except Exception as e:
        logger.error(f"Failed to generate report evaluation: {e}")
        return "报告生成完成。评价生成失败。"

def summarizer(state: State, config: RunnableConfig, **kwargs):
    logger.info("Starting summarizer_v2...")
    
    # 1. Reorganize content from tasks
    sections = _content_reorganizer(state)
    
    # 2. Generate Markdown content for each section using LLM
    full_markdown_content = []
    
    # Add Main Title
    report_title = find_title(state)
    full_markdown_content.append(f"# {report_title}\n")
    
    for section in sections:
        section_md = _generate_section_content(section, config)
        full_markdown_content.append(section_md)
        
    final_markdown = "\n".join(full_markdown_content)
    
    # 3. Save to session folder
    session_cache_dir = get_session_cache_dir(state, config)
    report_dir = os.path.join(session_cache_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    md_path = os.path.abspath(os.path.join(report_dir, "report.md"))
    rewritten_md_path = os.path.abspath(os.path.join(report_dir, "report_rewritten.md"))
    pdf_path = os.path.abspath(os.path.join(report_dir, "report.pdf"))
    docx_path = os.path.abspath(os.path.join(report_dir, "report.docx"))
    
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)
        logger.info(f"Markdown report saved to {md_path}")
    except Exception as e:
        logger.error(f"Failed to write markdown file: {e}")
        
    # 4. Convert to PDF, DOCX and Rewrite Markdown
    # Note: These utils might need to be imported or adjusted if they are not in the path or expect specific args
    
    # Generate Rewritten Markdown
    try:
        rewritten_content = md_rewrite.rewrite_markdown(final_markdown)
        with open(rewritten_md_path, "w", encoding="utf-8") as f:
            f.write(rewritten_content)
        logger.info(f"Rewritten Markdown report saved to {rewritten_md_path}")
    except Exception as e:
        logger.error(f"Failed to generate rewritten markdown: {e}")
    
    # Generate PDF
    try:
        math_img_dir = os.path.join(session_cache_dir, "math_imgs")
        md_to_pdf.md_to_pdf(final_markdown, pdf_path, math_img_dir=math_img_dir)
        logger.info(f"PDF report saved to {pdf_path}")
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        
    # Generate DOCX
    try:
        md_to_docx.md_to_docx(final_markdown, docx_path)
        logger.info(f"DOCX report saved to {docx_path}")
    except Exception as e:
        logger.error(f"Failed to generate DOCX: {e}")

    # 5. Return result
    # We return the paths to the generated files.
    
    # Generate Evaluation
    eval_text = _generate_report_evaluation(final_markdown, config)
    
    final_result = {
        "summary": "Report generated in Markdown, PDF, and DOCX formats.",
        "evaluation": eval_text,
        "attachments": [md_path, rewritten_md_path, pdf_path, docx_path],
        "path": docx_path, # Default to DOCX for compatibility
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    
    # We can also add a message to the conversation history
    msg_content = f"{eval_text}"
    msg = AIMessage(content=msg_content)
    
    return {"messages": [msg], "final_result": final_result}
