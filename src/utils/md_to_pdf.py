import os
import logging
import re
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, Image, ListFlowable, ListItem
from reportlab.graphics.shapes import Drawing, Line
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _split_markdown_table_row(line: str) -> list[str]:
    """Split one Markdown row without treating an escaped pipe as a boundary."""
    cells = []
    current = []
    text = line.strip()
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text) and text[index + 1] == "|":
            current.append("|")
            index += 2
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    cells.append("".join(current).strip())
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _normalize_table_rows(rows):
    """Pad ragged Markdown rows to the widest row."""
    width = max((len(row) for row in rows), default=0)
    return [list(row) + [""] * (width - len(row)) for row in rows]


def _is_table_divider(row: list[str]) -> bool:
    return bool(row) and all(
        re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
        for cell in row
    )

def register_chinese_font():
    """
    寻找并注册中文字体家族 (Normal, Bold, Italic)。
    返回注册的字体家族名称 (font_family_name)。
    """
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_fonts_dir = os.path.join(base_dir, "data", "fonts")
    
    # 1. 寻找常规字体 (Normal)
    normal_font_path = None
    normal_candidates = [
        os.path.join(data_fonts_dir, "MiSans-Normal.ttf"),
        # 系统字体回退
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    
    for path in normal_candidates:
        if os.path.exists(path):
            normal_font_path = path
            break
            
    if not normal_font_path:
        logger.warning("No Chinese font found. Using Helvetica.")
        return "Helvetica"

    # 2. 寻找粗体字体 (Bold)
    bold_font_path = None
    bold_subfont_index = 0
    # 优先尝试在 data/fonts 下找 MiSans-Bold
    bold_candidates = [
        (os.path.join(data_fonts_dir, "MiSans-Bold.ttf"), 0),
        # 系统字体回退
        (r"C:\Windows\Fonts\msyhbd.ttc", 0), # Windows System MSYH Bold
        (r"C:\Windows\Fonts\simhei.ttf", 0), # SimHei often used as Bold
    ]
    
    for path, index in bold_candidates:
        if os.path.exists(path):
            bold_font_path = path
            bold_subfont_index = index
            break
            
    # 如果没找到粗体，回退到常规字体
    if not bold_font_path:
        bold_font_path = normal_font_path

    # 3. 寻找斜体字体 (Italic) -> 使用 MiSans-Thin.ttf
    italic_font_path = None
    italic_candidates = [
        os.path.join(data_fonts_dir, "MiSans-Thin.ttf"),
    ]
    
    for path in italic_candidates:
        if os.path.exists(path):
            italic_font_path = path
            break
            
    # 如果没找到斜体，回退到常规字体
    if not italic_font_path:
        italic_font_path = normal_font_path
        
    family_name = "ChineseFont"
    
    try:
        # 注册 Regular
        # 注意：为了让 ParagraphStyle(fontName=family_name) 工作，
        # 我们必须注册一个名字就是 family_name 的字体作为 Normal，或者确保映射正确。
        # 最简单的做法：Regular 字体直接用 Family Name 注册。
        pdfmetrics.registerFont(TTFont(family_name, normal_font_path))
        
        # 注册 Bold
        # 如果是 TTC 且 index > 0，需要指定 subfontIndex
        if bold_font_path.lower().endswith('.ttc'):
             pdfmetrics.registerFont(TTFont(f'{family_name}-Bold', bold_font_path, subfontIndex=bold_subfont_index))
        else:
             pdfmetrics.registerFont(TTFont(f'{family_name}-Bold', bold_font_path))
             
        # 注册 Italic (使用 Thin 或 fallback)
        pdfmetrics.registerFont(TTFont(f'{family_name}-Italic', italic_font_path))
        
        # 注册 BoldItalic (复用 Bold)
        pdfmetrics.registerFont(TTFont(f'{family_name}-BoldItalic', bold_font_path if bold_font_path.lower().endswith('.ttf') else normal_font_path))
        
        # 注册家族映射
        pdfmetrics.registerFontFamily(
            family_name,
            normal=family_name, # Normal 指向自身
            bold=f'{family_name}-Bold',
            italic=f'{family_name}-Italic',
            boldItalic=f'{family_name}-BoldItalic'
        )
        
        logger.info(f"Successfully registered font family '{family_name}'")
        logger.info(f"  Normal: {normal_font_path}")
        logger.info(f"  Bold:   {bold_font_path} (Index {bold_subfont_index})")
        logger.info(f"  Italic: {italic_font_path}")
        
    except Exception as e:
        logger.warning(f"Font registration failed: {e}")
        return "Helvetica"
        
    return family_name

def render_math_to_image(latex_str, fontsize=14, dpi=300):
    """
    使用 Matplotlib 将 LaTeX 公式渲染为图片 (BytesIO)
    """
    fig = None
    try:
        # 创建 Figure
        # 尺寸设得非常小，依赖 bbox_inches='tight' 扩展
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${latex_str}$", fontsize=fontsize)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        if fig:
            plt.close(fig)
        logger.warning(f"Math render failed for '{latex_str}': {e}")
        return None

def parse_markdown(content, styles, font_name, math_img_dir=None):
    """
    简单的 Markdown 解析器，将文本转换为 ReportLab Flowables。
    支持：
    - 标题 (#, ##...)
    - 列表 (-, *, 1.)
    - 代码块 (```)
    - 简单表格 (|...|)
    - 图片 (![alt](src))
    - 粗体/斜体 (**...**, *...*)
    """
    story = []
    lines = content.split('\n')
    
    # 预定义样式
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        spaceAfter=6
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontName=font_name, # 如果注册了家族，Bold 会自动生效吗？
        # ReportLab Heading 样式默认可能没有 bold=True，需要检查
        # 这里的 fontName 是家族名。如果样式指定了 fontName='ChineseFont'，
        # 要让它显示粗体，需要 style.fontName 指向 Bold 字体名，或者使用 <b> 标签。
        # 标准做法：Heading1 默认通常是 Bold。
        # 如果我们注册了 Family，并且 style 设置了 fontName='ChineseFont'，
        # 那么 <b> 标签会切换到 'ChineseFont-Bold'。
        # 但整个段落加粗需要设置 style.fontName = 'ChineseFont-Bold' 或者使用 <b> 包裹内容。
        fontSize=18,
        leading=22,
        spaceAfter=12
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        leading=18,
        spaceAfter=10
    )

    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontName=font_name,
        fontSize=12,
        leading=16,
        spaceAfter=8
    )

    h4_style = ParagraphStyle(
        'CustomH4',
        parent=styles['Heading4'],
        fontName=font_name,
        fontSize=11,
        leading=14,
        spaceAfter=6,
        textColor=colors.black
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontName=font_name, 
        fontSize=8,
        leading=10,
        textColor=colors.black,
        backColor=colors.lightgrey,
        borderPadding=5
    )
    
    # 图注样式 (居中)
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        leading=11,
        alignment=1, # 1=CENTER
        spaceAfter=6,
        textColor=colors.dimgrey
    )
    
    # 状态机
    in_code_block = False
    code_buffer = []
    
    # 标题自动编号堆栈
    # 规则：# 不编号，## 开始编号 (Index 0)
    heading_stack = []
    
    in_table = False
    table_buffer = []
    table_caption = None # 表格标题
    
    # Description Tag State
    in_description = False
    description_buffer = []
    pending_table_caption = None
    
    # Math Block State
    in_math_block = False
    math_buffer = []

    for line in lines:
        stripped_line = line.strip()
        
        # 1. 处理代码块
        if stripped_line.startswith('```'):
            if in_code_block:
                # 结束代码块
                in_code_block = False
                code_text = '\n'.join(code_buffer)
                story.append(Preformatted(code_text, code_style))
                code_buffer = []
            else:
                if in_table:
                    _flush_table(story, table_buffer, font_name, table_caption, caption_style)
                    in_table = False
                    table_buffer = []
                    table_caption = None
                in_code_block = True
            continue
        
        if in_code_block:
            code_buffer.append(line)
            continue

        # 1.5 处理公式块 ($$ ... $$)
        if stripped_line.startswith('$$'):
            if stripped_line == '$$':
                if in_math_block:
                    # End block
                    in_math_block = False
                    tex = '\n'.join(math_buffer)
                    math_buffer = []
                    
                    img_buf = render_math_to_image(tex)
                    if img_buf:
                        img = Image(img_buf)
                        # Scaling: 300 dpi -> 72 dpi
                        scale_factor = 72.0 / 300.0
                        img.drawWidth = img.imageWidth * scale_factor
                        img.drawHeight = img.imageHeight * scale_factor
                        
                        story.append(img)
                        story.append(Spacer(1, 6))
                    else:
                        story.append(Paragraph(f"[Math Error]", code_style))
                else:
                    # Start block
                    in_math_block = True
                    math_buffer = []
                continue
            elif stripped_line.endswith('$$') and len(stripped_line) > 2:
                # Single line block $$...$$
                tex = stripped_line[2:-2]
                img_buf = render_math_to_image(tex)
                if img_buf:
                    img = Image(img_buf)
                    scale_factor = 72.0 / 300.0
                    img.drawWidth = img.imageWidth * scale_factor
                    img.drawHeight = img.imageHeight * scale_factor
                    story.append(img)
                    story.append(Spacer(1, 6))
                continue
        
        if in_math_block:
            math_buffer.append(line)
            continue

        # 2. 处理 <description> 标签 (图注/表注)
        # 必须在处理 Image/Table 之前处理，也必须在 Code 之后
        # 检查是否包含标签
        full_desc = ""
        if '<description>' in stripped_line or in_description:
            # 如果是起始行
            if '<description>' in stripped_line:
                in_description = True
                # 提取开始部分
                start_idx = line.find('<description>') + len('<description>')
                # 检查是否同在一行结束
                if '</description>' in line:
                    end_idx = line.find('</description>')
                    content = line[start_idx:end_idx].strip()
                    
                    # 立即处理
                    full_desc = content
                    in_description = False
                    description_buffer = []
                else:
                    # 多行开始
                    content = line[start_idx:].strip()
                    if content:
                        description_buffer.append(content)
                    continue # 继续下一行寻找结束
            
            elif in_description:
                # 寻找结束标签
                if '</description>' in stripped_line:
                    end_idx = line.find('</description>')
                    content = line[:end_idx].strip()
                    if content:
                        description_buffer.append(content)
                    
                    full_desc = " ".join(description_buffer).strip()
                    in_description = False
                    description_buffer = []
                else:
                    # 中间行
                    description_buffer.append(stripped_line)
                    continue

            # 处理提取到的 full_desc
            if not in_description: # 说明刚刚结束
                # 逻辑：
                # 1. 检查上一元素是否为图片 (Image 或 Image+Spacer)
                last_is_image = False
                if story:
                    if isinstance(story[-1], Image):
                        last_is_image = True
                    elif len(story) > 1 and isinstance(story[-2], Image) and isinstance(story[-1], Spacer):
                        last_is_image = True
                
                if last_is_image:
                    # 解析为图注
                    story.append(Paragraph(f"<i>图: {full_desc}</i>", caption_style))
                    # 清除可能存在的 pending (以防万一)
                    pending_table_caption = None
                else:
                    # 解析为表格标题 (Pending)
                    pending_table_caption = full_desc
            
            # 如果当前行不仅包含 description，还包含其他内容？
            # 假设 description 独占一行或块。
            # 如果是单行 <description>...</description>，处理完 continue
            if '<description>' in stripped_line and '</description>' in stripped_line:
                continue
            if '</description>' in stripped_line:
                continue
        
        # 2.5 处理分隔符 (---)
        if re.match(r'^[-*_]{3,}$', stripped_line):
            # 添加一条水平线
            # A4 width ~595 pts. Margins 72+72=144. Content width ~450.
            width = 450
            d = Drawing(width, 10)
            d.add(Line(0, 5, width, 5, strokeColor=colors.grey, strokeWidth=1))
            story.append(Spacer(1, 6))
            story.append(d)
            story.append(Spacer(1, 6))
            continue

        # 3. 处理表格
        # ... (Existing logic modified to use pending_table_caption)
        if stripped_line.startswith('|') and stripped_line.endswith('|'):
            if not in_table:
                # 优先使用 <description> 提供的标题
                if pending_table_caption:
                     table_caption = Paragraph(f"<b>{pending_table_caption}</b>", caption_style)
                     pending_table_caption = None
                else:
                    # 尝试从 story 中提取表注 (旧逻辑保持兼容，或移除)
                    if story and isinstance(story[-1], Paragraph):
                        last_para_text = story[-1].text
                        if re.match(r'^(表|Table)\s*\d+[:：]', last_para_text, re.IGNORECASE):
                            table_caption = story.pop()
                            table_caption = Paragraph(last_para_text, caption_style)
                in_table = True
            
            row_data = _split_markdown_table_row(line)
            if row_data and not _is_table_divider(row_data):
                table_buffer.append(row_data)
            continue
        else:
            if in_table:
                _flush_table(story, table_buffer, font_name, table_caption, caption_style)
                in_table = False
                table_buffer = []
                table_caption = None
        
        if not stripped_line:
            continue
            
        # 3. 处理图片 (![alt](src))
        img_match = re.match(r'!\[(.*?)\]\((.*?)\)', stripped_line)
        if img_match:
            alt_text = img_match.group(1)
            img_path = img_match.group(2)
            
            # 尝试加载图片
            if os.path.exists(img_path):
                try:
                    # 限制图片宽度
                    # A4 width ~595 pts. Margins 72+72=144. Content width ~450.
                    img = Image(img_path)
                    
                    # 简单的缩放逻辑
                    max_width = 450
                    if img.drawWidth > max_width:
                        ratio = max_width / img.drawWidth
                        img.drawWidth = max_width
                        img.drawHeight = img.drawHeight * ratio
                        
                    story.append(img)
                    story.append(Spacer(1, 6))
                    if alt_text:
                        # 图注居中
                        story.append(Paragraph(f"<i>图: {alt_text}</i>", caption_style))
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    story.append(Paragraph(f"[Image load failed: {img_path}]", normal_style))
            else:
                story.append(Paragraph(f"[Image not found: {img_path}]", normal_style))
            continue
            
        # 4. 标题
        if stripped_line.startswith('# '):
            # Heading1 内容加粗，作为文档标题，不编号，但重置编号堆栈
            heading_stack = []
            text = f"<b>{stripped_line[2:]}</b>"
            story.append(Paragraph(text, h1_style))
        elif stripped_line.startswith('## '):
            # Level 2 -> Index 0 (e.g. 1.)
            while len(heading_stack) <= 0:
                heading_stack.append(0)
            heading_stack[0] += 1
            heading_stack = heading_stack[:1]
            
            num_str = f"{heading_stack[0]}."
            text = f"<b>{num_str} {stripped_line[3:]}</b>"
            story.append(Paragraph(text, h2_style))
        elif stripped_line.startswith('### '):
            # Level 3 -> Index 1 (e.g. 1.1)
            while len(heading_stack) <= 1:
                heading_stack.append(0)
            heading_stack[1] += 1
            heading_stack = heading_stack[:2]
            
            num_str = f"{heading_stack[0]}.{heading_stack[1]}"
            text = f"<b>{num_str} {stripped_line[4:]}</b>"
            story.append(Paragraph(text, h3_style))
        elif stripped_line.startswith('#### '):
            # Level 4 -> Index 2 (e.g. 1.1.1)
            while len(heading_stack) <= 2:
                heading_stack.append(0)
            heading_stack[2] += 1
            heading_stack = heading_stack[:3]
            
            num_str = f"{heading_stack[0]}.{heading_stack[1]}.{heading_stack[2]}"
            text = f"<b>{num_str} {stripped_line[5:]}</b>"
            story.append(Paragraph(text, h4_style))
            
        # 5. 无序列表
        elif stripped_line.startswith('- ') or stripped_line.startswith('* '):
            text = _format_text(stripped_line[2:], math_img_dir=math_img_dir)
            # 使用 ListItem 可能更规范，但这里用 bullet char 简单模拟
            story.append(Paragraph(f"• {text}", normal_style))
            
        # 6. 有序列表 (1. item)
        elif re.match(r'^\d+\.\s', stripped_line):
            match = re.match(r'^(\d+)\.\s+(.*)', stripped_line)
            num = match.group(1)
            content = _format_text(match.group(2), math_img_dir=math_img_dir)
            story.append(Paragraph(f"{num}. {content}", normal_style))
            
        # 7. 普通文本
        else:
            formatted_text = _format_text(stripped_line, math_img_dir=math_img_dir)
            story.append(Paragraph(formatted_text, normal_style))
            
    if in_table and table_buffer:
        _flush_table(story, table_buffer, font_name, table_caption, caption_style)
        
    return story

def _process_inline_math(text, math_img_dir=None):
    """
    处理行内公式 $...$
    将其转换为 <img ... /> 标签嵌入 Paragraph
    """
    pattern = re.compile(r'(?<!\\)\$(.+?)(?<!\\)\$')
    
    def repl(match):
        content = match.group(1)
        # Verify it's not empty
        if not content.strip():
            return match.group(0)
            
        # 缓存机制：基于内容哈希
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        img_dir = os.path.abspath(math_img_dir or "./tmp/math_imgs")
        img_path = os.path.join(img_dir, f"{content_hash}.png")
        
        # 渲染并保存
        if not os.path.exists(img_path):
             os.makedirs(img_dir, exist_ok=True)
             # 使用稍大的字号以保证清晰度，后续缩小显示
             img_buf = render_math_to_image(content, fontsize=14, dpi=300)
             if not img_buf:
                 return f"${content}$" # Failed
             
             with open(img_path, 'wb') as f:
                 f.write(img_buf.getvalue())
        
        # 计算显示尺寸
        try:
            with PILImage.open(img_path) as img:
                w, h = img.size
                
            # 目标高度：与正文文字高度接近 (e.g. 10pt)
            # 10 pt = 10/72 inch
            # h pixels = h/300 inch
            # scale = target / source
            
            target_h = 10
            aspect = w / h
            target_w = target_h * aspect
            
            # valign: 垂直偏移，负值向下
            # 稍微向下偏移以对齐基线
            valign = -2
            
            # ReportLab 的 img 标签路径建议使用 forward slashes
            safe_path = img_path.replace('\\', '/')
            
            return f'<img src="{safe_path}" width="{target_w}" height="{target_h}" valign="{valign}"/>'
        except Exception as e:
            logger.error(f"Error processing inline math image: {e}")
            return f"${content}$"

    return pattern.sub(repl, text)

def _format_text(text, math_img_dir=None):
    """处理行内格式：粗体、斜体、行内公式"""
    # 0. 行内公式 (优先处理，避免 * 等符号冲突)
    text = _process_inline_math(text, math_img_dir=math_img_dir)

    # 1. 粗斜体 ***text*** -> <b><i>text</i></b>
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', text)
    # 2. 粗体 **text** -> <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # 3. 斜体 *text* -> <i>text</i>
    # 注意：要避免匹配到 ** 的一部分。由于上面已经替换了 **，这里匹配 * 应该比较安全
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    return text


def _flush_table(
    story,
    table_buffer,
    font_name,
    table_caption=None,
    caption_style=None,
    available_width=451.0,
):
    """辅助函数：将缓冲的表格数据添加到 Story"""
    if not table_buffer:
        return
        
    # 添加表注 (如果存在)
    if table_caption:
        story.append(table_caption)
        story.append(Spacer(1, 2))
        
    rows = _normalize_table_rows(table_buffer)
    column_count = len(rows[0]) if rows else 0
    if column_count == 0:
        return
    font_size = max(6.0, min(10.0, 60.0 / column_count))

    # 创建表格
    # 将每个单元格内容包装为 Paragraph 以支持换行和字体
    data = []
    cell_style = ParagraphStyle(
        'CellStyle',
        fontName=font_name,
        fontSize=font_size,
        leading=max(7.0, font_size + 2.0),
    )
    
    for row in rows:
        processed_row = [Paragraph(cell, cell_style) for cell in row]
        data.append(processed_row)
        
    column_width = available_width / column_count
    t = Table(
        data,
        colWidths=[column_width] * column_count,
        repeatRows=1,
    )
    
    # 表格样式
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), font_name), # 表头字体
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('BOTTOMPADDING', (0, 0), (-1, 0), max(4, font_size)),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), font_name), # 全局字体
    ]))
    
    story.append(t)
    story.append(Spacer(1, 12))

def md_to_pdf(content: str, output_path: str, math_img_dir: str = None) -> str:
    """
    将 Markdown 转换为 PDF (使用 ReportLab).
    """
    # 1. 注册字体
    font_name = register_chinese_font()
    
    # 2. 准备文档
    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        abs_output_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # 3. 解析内容
    styles = getSampleStyleSheet()
    story = parse_markdown(content, styles, font_name, math_img_dir=math_img_dir)
    
    # 4. 生成 PDF
    try:
        doc.build(story)
        logger.info(f"PDF generated successfully at {abs_output_path}")
    except Exception as e:
        logger.error(f"Failed to build PDF: {e}")
        raise
        
    return abs_output_path

if __name__ == "__main__":
    # Test
    # 确保测试图片存在 (由 create_test_image.py 生成或手动放置)
    # 假设 d:/Agent/test_image.png
    
    md = r"""# Markdown 转 PDF 测试报告
    
## 字体测试
普通文本：这是 MiSans Normal 字体（在 DOCX 中应为微软雅黑）。
**粗体文本：这是 Bold 字体。**
*斜体文本：这是 Italic 字体。*
***粗斜体文本：Bold + Italic。***

## 列表测试
### 无序列表
- 项目 A
- 项目 B

### 有序列表
1. 第一步：打开冰箱
2. 第二步：把大象装进去
3. 第三步：关上冰箱

## 表格测试
<description>表1: 员工信息表 (PDF Test)</description>
| 姓名 | 年龄 | 职业 | 备注 |
| --- | --- | --- | --- |
| 张三 | 25 | 工程师 | **优秀** |
| 李四 | 30 | 设计师 | *骨干* |
| 王五 | 28 | 产品经理 | 普通 |

## 图片测试
![AltIgnored](test_image.png)
<description>图1: 测试图片-居中图注 (PDF Test)</description>

## 代码测试
```
def hello_world():
    print("Hello, Agent!")
    return True
```

## 分隔符测试
---
(上面应该有一条线)

## 公式测试
下面是一个公式块：
$$
E = mc^2
$$

复杂公式：
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

行内公式块样式（独立成行）：
$$ F = ma $$

行内嵌入测试：
这是一个嵌入在文字中间的公式 $E=mc^2$ ，应该和文字在同一行。
再来一个稍微复杂点的： $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$ ，看看对齐效果。
如果不加空格$a^2+b^2=c^2$也是可以的。
测试转义 \$NotMath\$。
"""
    path = "./test_pdf_full.pdf"
    print(f"Generating PDF at {path}...")
    try:
        # 确保图片路径正确 (相对于运行目录)
        # 如果我们在 D:\Agent 下运行，test_image.png 就在当前目录
        result = md_to_pdf(md, path)
        print(f"Success! Saved to: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
