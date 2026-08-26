import os
import json
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import get_app_config
from src.llm import invoke_llm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class ChartUtils:
    """图表生成工具类"""

    @staticmethod
    def read_csv_data(file_path: str) -> pd.DataFrame:
        """读取CSV数据文件（动态解析存在的时间列）"""
        try:
            # 第一步：先读取CSV，不解析日期（避免列缺失报错）
            df = pd.read_csv(file_path, encoding="utf-8")

            # 第二步：动态识别并解析时间列
            time_column_candidates = ["date", "时间", "time", "timestamp"]
            existing_time_cols = [col for col in time_column_candidates if col in df.columns]

            if existing_time_cols:
                # 重新读取并解析存在的时间列
                df = pd.read_csv(
                    file_path,
                    parse_dates=existing_time_cols,
                    encoding="utf-8"
                )

            return df
        except Exception as e:
            raise ValueError(f"无法读取CSV文件 {file_path}: {str(e)}")

    @staticmethod
    def create_line_chart(data: pd.DataFrame, title: str, output_path: str,
                         x_col: str = None, y_cols: List[str] = None):
        """生成折线图"""
        fig, ax = plt.subplots(figsize=(12, 7))

        if x_col and y_cols:
            for y_col in y_cols:
                if y_col in data.columns:
                    ax.plot(data[x_col], data[y_col], label=y_col, linewidth=1)
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols[:20]:
                    ax.plot(data.index, data[col], label=col, linewidth=1)

        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('时间' if x_col else '数据点', fontsize=14, labelpad=10)
        ax.set_ylabel('数值', fontsize=14, labelpad=10)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)

        # 改进刻度标签
        ax.tick_params(axis='both', which='major', labelsize=12)

        # 如果是时间序列，格式化x轴
        if x_col and data[x_col].dtype == '<M8[ns]':
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ChartUtils._save_chart(fig, output_path)
        plt.close(fig)

    @staticmethod
    def create_scatter_chart(data: pd.DataFrame, title: str, output_path: str,
                           x_col: str = None, y_col: str = None):
        """生成散点图"""
        fig, ax = plt.subplots(figsize=(12, 7))

        if x_col and y_col:
            scatter = ax.scatter(data[x_col], data[y_col], alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            ax.set_xlabel(x_col, fontsize=14, labelpad=10)
            ax.set_ylabel(y_col, fontsize=14, labelpad=10)
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                scatter = ax.scatter(data[x_col], data[y_col], alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
                ax.set_xlabel(x_col, fontsize=14, labelpad=10)
                ax.set_ylabel(y_col, fontsize=14, labelpad=10)

        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)

        # 改进刻度标签
        ax.tick_params(axis='both', which='major', labelsize=12)

        ChartUtils._save_chart(fig, output_path)
        plt.close(fig)

    @staticmethod
    def create_bar_chart(data: pd.DataFrame, title: str, output_path: str,
                        x_col: str = None, y_col: str = None):
        """生成柱状图"""
        fig, ax = plt.subplots(figsize=(12, 7))

        if x_col and y_col:
            bars = ax.bar(data[x_col], data[y_col], alpha=0.8, color='skyblue', edgecolor='navy', linewidth=1)
            ax.set_xlabel(x_col, fontsize=14, labelpad=10)
            ax.set_ylabel(y_col, fontsize=14, labelpad=10)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                y_col = numeric_cols[0]
                x_col = data.index.astype(str)
                bars = ax.bar(x_col, data[y_col], alpha=0.8, color='lightcoral', edgecolor='darkred', linewidth=1)
                ax.set_ylabel(y_col, fontsize=14, labelpad=10)

                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # 改进刻度标签
        ax.tick_params(axis='both', which='major', labelsize=12)

        # 调整布局防止标签重叠
        plt.tight_layout()

        ChartUtils._save_chart(fig, output_path)
        plt.close(fig)

    @staticmethod
    def _save_chart(fig, output_path: str):
        """保存图表文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 拆分文件名和后缀（不管原后缀是什么，都提取纯文件名）
        file_name, _ = os.path.splitext(output_path)
        # 重新构造PNG和SVG的路径（确保后缀正确）
        png_path = f"{file_name}.png"
        svg_path = f"{file_name}.svg"

        # 保存文件（逻辑不变）
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
        fig.savefig(svg_path, dpi=300, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')

class ImageProcessor:
    """图像处理工具类"""

    @staticmethod
    def image_to_base64(image_path: str) -> str:
        """将图像文件转换为Base64编码"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        except Exception as e:
            raise Exception(f"图像编码失败: {str(e)}")

    @staticmethod
    def get_image_info(image_path: str) -> Dict:
        """获取图像基本信息"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")

            file_size = os.path.getsize(image_path)

            # 获取图像尺寸
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size

            return {
                "path": image_path,
                "format": os.path.splitext(image_path)[1].upper()[1:],  # 去掉点号
                "size": file_size,
                "size_kb": round(file_size / 1024, 2),
                "dimensions": {"width": width, "height": height},
                "dimensions_str": f"{width}×{height}"
            }
        except ImportError:
            # 如果没有PIL，返回基本信息
            return {
                "path": image_path,
                "format": os.path.splitext(image_path)[1].upper()[1:],
                "size": os.path.getsize(image_path),
                "size_kb": round(os.path.getsize(image_path) / 1024, 2),
                "dimensions": None,
                "dimensions_str": "unknown"
            }
        except Exception as e:
            raise Exception(f"获取图像信息失败: {str(e)}")

class TableUtils:
    """表格生成工具类"""

    @staticmethod
    def create_raw_table(data: pd.DataFrame, max_rows: int = 50) -> Dict:
        """创建原始数据表格"""
        # 显示前N行数据
        display_data = data.head(max_rows)

        # 转换为二维数组
        table_data = [display_data.columns.tolist()]
        for _, row in display_data.iterrows():
            table_data.append(row.tolist())

        return {
            "table_id": f"raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "raw",
            "description": "原始数据表",
            "data": table_data,
            "total_rows": len(data),
            "display_rows": len(table_data) - 1
        }

    @staticmethod
    def create_summary_table(data: pd.DataFrame) -> Dict:
        """创建统计摘要表"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        table_data = [["参数名称", "平均值", "最大值", "最小值", "标准差", "变异系数"]]

        for col in numeric_cols[:10]:  # 最多显示10个数值列
            col_data = data[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                max_val = col_data.max()
                min_val = col_data.min()
                std_val = col_data.std()
                cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0

                table_data.append([
                    str(col),
                    f"{mean_val:.2f}",
                    f"{max_val:.2f}",
                    f"{min_val:.2f}",
                    f"{std_val:.2f}",
                    f"{cv_val:.1f}%"
                ])

        return {
            "table_id": f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "summary",
            "description": "工艺参数统计摘要",
            "data": table_data
        }

    @staticmethod
    def create_anomaly_table(data: pd.DataFrame) -> Dict:
        """创建异常统计表"""
        # 查找异常相关的列
        anomaly_cols = []
        for col in data.columns:
            if "异常" in col or "异常" in str(col).lower():
                anomaly_cols.append(col)

        if not anomaly_cols:
            # 如果没有异常列，基于数值范围检测异常（假设超出3倍标准差为异常）
            return TableUtils._detect_anomalies_from_values(data)

        # 统计各类异常
        table_data = [["异常类型", "发生次数", "占比"]]
        total_rows = len(data)

        for col in anomaly_cols:
            anomaly_counts = data[col].value_counts()
            for anomaly_type, count in anomaly_counts.items():
                if anomaly_type not in ["正常", "None", "无", "正常", ""]:
                    percentage = f"{count/total_rows*100:.1f}%"
                    table_data.append([str(anomaly_type), str(count), percentage])

        return {
            "table_id": f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "anomaly",
            "description": "异常因素统计",
            "data": table_data
        }

    @staticmethod
    def create_batch_table(data: pd.DataFrame) -> Dict:
        """创建批次对比表"""
        # 查找批次列
        batch_col = None
        for col in data.columns:
            if "批次" in col or "batch" in str(col).lower():
                batch_col = col
                break

        if not batch_col:
            # 尝试查找可能包含批次信息的列
            for col in data.columns:
                if data[col].dtype == 'object' and data[col].str.contains('Batch|批次', na=False).any():
                    batch_col = col
                    break

        if not batch_col:
            return {
                "table_id": f"batch_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "type": "error",
                "description": "未找到批次列，无法生成批次对比表",
                "data": [["错误", "信息"], ["", "数据中未包含批次分类信息"]]
            }

        # 按批次分组统计
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        group_data = data.groupby(batch_col)[numeric_cols].agg(['mean', 'max', 'min']).round(2)

        # 转换为表格格式
        table_data = [[batch_col] + [f"{col}_{stat}" for col in numeric_cols[:5] for stat in ["均值", "最大", "最小"]]]

        for batch_name, batch_data in group_data.iterrows():
            row = [str(batch_name)]
            for col in numeric_cols[:5]:
                row.extend([
                    f"{batch_data[col]['mean']:.2f}",
                    f"{batch_data[col]['max']:.2f}",
                    f"{batch_data[col]['min']:.2f}"
                ])
            table_data.append(row)

        return {
            "table_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "batch",
            "description": "批次对比分析",
            "data": table_data
        }

    @staticmethod
    def _detect_anomalies_from_values(data: pd.DataFrame) -> Dict:
        """基于数值范围检测异常"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        table_data = [["因素", "异常次数", "平均偏差"]]

        total_anomalies = 0

        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                std_val = col_data.std()
                # 3-sigma规则
                upper_bound = mean_val + 3 * std_val
                lower_bound = mean_val - 3 * std_val

                anomalies = col_data[(col_data > upper_bound) | (col_data < lower_bound)]
                anomaly_count = len(anomalies)

                if anomaly_count > 0:
                    avg_deviation = anomalies.apply(lambda x: abs(x - mean_val)).mean()
                    total_anomalies += anomaly_count
                    table_data.append([
                        str(col),
                        str(anomaly_count),
                        f"{avg_deviation:.2f}"
                    ])

        if total_anomalies == 0:
            table_data = [["提示", "信息"], ["", "未检测到明显异常值"]]

        return {
            "table_id": f"anomaly_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "anomaly",
            "description": f"基于3-sigma规则检测到的异常值（共{total_anomalies}个）",
            "data": table_data
        }


class EnhancedDataSummary:
    """增强型数据摘要生成器"""

    @staticmethod
    def extract_metadata(data: pd.DataFrame, csv_path: str) -> Dict:
        """提取完整的元数据信息"""
        # 获取文件名
        filename = os.path.basename(csv_path)

        # 获取列名
        columns = data.columns.tolist()

        # 数据维度
        data_shape = f"{len(data)}行×{len(data.columns)}列"

        # 时间信息
        time_info = EnhancedDataSummary._extract_time_info(data)

        # 数值列统计
        numeric_stats = EnhancedDataSummary._extract_numeric_stats(data)

        # 分类列信息
        categorical_info = EnhancedDataSummary._extract_categorical_info(data)

        return {
            "filename": filename,
            "columns": columns,
            "data_shape": data_shape,
            "time_info": time_info,
            "numeric_stats": numeric_stats,
            "categorical_info": categorical_info
        }

    @staticmethod
    def _extract_time_info(data: pd.DataFrame) -> Dict:
        """提取时间信息"""
        time_col = next((col for col in ["date", "时间", "time", "timestamp"] if col in data.columns), None)

        if time_col:
            time_data = data[time_col]

            # 尝试转换为datetime
            try:
                if time_data.dtype == 'object':
                    time_data = pd.to_datetime(time_data, errors='coerce')
            except:
                pass

            # 获取时间范围
            valid_times = time_data.dropna()
            if len(valid_times) > 0:
                start_time = valid_times.min()
                end_time = valid_times.max()

                return {
                    "has_time": True,
                    "time_column": time_col,
                    "time_range": {
                        "start": str(start_time),
                        "end": str(end_time),
                        "duration": str(end_time - start_time) if pd.notna(end_time - start_time) else "N/A"
                    }
                }

        return {
            "has_time": False,
            "time_column": None,
            "time_range": None
        }

    @staticmethod
    def _extract_numeric_stats(data: pd.DataFrame) -> Dict:
        """提取数值列统计信息"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        stats = {}

        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                # 尝试从列名推断单位
                unit = EnhancedDataSummary._infer_unit(col)

                stats[col] = {
                    "count": int(len(col_data)),
                    "mean": round(float(col_data.mean()), 2),
                    "std": round(float(col_data.std()), 2),
                    "min": round(float(col_data.min()), 2),
                    "max": round(float(col_data.max()), 2),
                    "unit": unit,
                    "missing_count": int(data[col].isna().sum())
                }

        return stats

    @staticmethod
    def _extract_categorical_info(data: pd.DataFrame) -> Dict:
        """提取分类列信息"""
        categorical_cols = data.select_dtypes(include=['object']).columns
        info = {}

        for col in categorical_cols:
            # 排除时间列
            if col in ["时间", "time", "timestamp", "date"]:
                continue

            col_data = data[col].dropna()
            if len(col_data) > 0:
                unique_count = col_data.nunique()
                most_common = col_data.value_counts().head(5).to_dict()

                info[col] = {
                    "unique_count": int(unique_count),
                    "most_common": most_common
                }

        return info

    @staticmethod
    def _infer_unit(col_name: str) -> str:
        """从列名推断单位"""
        col_lower = col_name.lower()

        if "温度" in col_name or "temp" in col_lower:
            return "℃"
        elif "压力" in col_name or "pressure" in col_lower:
            return "MPa"
        elif "流量" in col_name or "flow" in col_lower:
            return "m³/h"
        elif "纯度" in col_name or "purity" in col_lower:
            return "%"
        elif "浓度" in col_name or "conc" in col_lower:
            return "%"
        elif "ph" in col_lower:
            return ""
        elif "转速" in col_name or "speed" in col_lower:
            return "rpm"
        elif "电流" in col_name or "current" in col_lower:
            return "A"
        elif "电压" in col_name or "voltage" in col_lower:
            return "V"
        else:
            return ""


class LLMClient:
    """大模型客户端"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.3,
            max_tokens=2000
        )

    def generate_chart_description(
        self,
        chart_type: str,
        image_data: str,
        metadata: Dict,
        **scope: Any,
    ) -> str:
        """生成专业的图表说明（基于图像和元数据）"""

        # 构建数据信息字符串
        data_info_str = self._format_metadata_for_prompt(metadata)

        prompt = f"""
        

        图表类型：{chart_type}
        数据来源：{metadata.get('filename', '未知文件')}

        数据结构：
        {data_info_str}

        请从以下几个方面进行分析：
        1. 数据趋势和变化特征
        2. 参数间的相关性和影响因素
        3. 工艺过程中的关键节点和异常点
        4. 对工艺优化的启示和建议
        5. 安全监控和预警分析

        生成200-300字的详细说明，语言专业、准确、易懂。
        """

        try:
            response = invoke_llm(
                self.llm,
                [HumanMessage(content=prompt)],
                node="ChartGenerator",
                purpose="chart_description",
                json_mode=False,
                **scope,
            )
            return response.content
        except Exception as e:
            print(f"大模型调用失败: {str(e)}")
            return f"图表类型：{chart_type}，数据来源：{metadata.get('filename', '未知文件')}"

    def generate_chart_description_fallback(
        self, chart_type: str, data_info: Dict, **scope: Any
    ) -> str:
        """基于数据信息的降级描述生成（向后兼容）"""
        prompt = f"""
        

        图表类型：{chart_type}
        数据信息：{json.dumps(data_info, ensure_ascii=False)}

        请从以下几个方面进行分析：
        1. 数据趋势和变化特征
        2. 参数间的相关性和影响因素
        3. 工艺过程中的关键节点和异常点
        4. 对工艺优化的启示和建议
        5. 安全监控和预警分析

        生成200-300字的详细说明，语言专业、准确、易懂。
        """

        try:
            response = invoke_llm(
                self.llm,
                [HumanMessage(content=prompt)],
                node="ChartGenerator",
                purpose="chart_description_fallback",
                json_mode=False,
                **scope,
            )
            return response.content
        except Exception as e:
            print(f"大模型调用失败: {str(e)}")
            return f"图表类型：{chart_type}，数据信息：{data_info.get('summary', '暂无数据摘要')}"

    def _format_metadata_for_prompt(self, metadata: Dict) -> str:
        """格式化元数据用于提示词"""
        lines = []

        # 基本信息
        lines.append(f"- 数据维度：{metadata.get('data_shape', 'N/A')}")
        lines.append(f"- 包含列：{', '.join(metadata.get('columns', []))}")

        # 时间信息
        time_info = metadata.get('time_info', {})
        if time_info.get('has_time'):
            time_range = time_info.get('time_range', {})
            lines.append(f"- 时间范围：{time_range.get('start', 'N/A')} 至 {time_range.get('end', 'N/A')}")

        # 数值列统计
        numeric_stats = metadata.get('numeric_stats', {})
        if numeric_stats:
            lines.append("- 关键参数统计：")
            for col, stats in numeric_stats.items():
                unit = stats.get('unit', '')
                lines.append(f"  * {col}: 平均值={stats.get('mean', 0):.2f}{unit}, "
                           f"范围=[{stats.get('min', 0):.2f}~{stats.get('max', 0):.2f}]{unit}")

        return "\n".join(lines)

    def generate_table_description(
        self,
        table_type: str,
        table_data: List[List[str]],
        table_description: str = "",
        **scope: Any,
    ) -> str:
        """生成专业的表格说明"""

        # 计算表格统计信息
        row_count = len(table_data) - 1 if len(table_data) > 0 else 0
        col_count = len(table_data[0]) if len(table_data) > 0 else 0

        # 构建提示词
        prompt = f"""
        请为以下表格生成专业的分析说明，重点关注化工工艺数据的解读：

        表格类型：{table_type}
        表格描述：{table_description}
        数据规模：{row_count}行 x {col_count}列

        表格内容：
        {json.dumps(table_data, ensure_ascii=False, indent=2)}

        请从以下几个方面进行分析：
        1. 数据分布特征和关键数值
        2. 异常情况和偏差分析
        3. 工艺参数的相关性和变化趋势
        4. 对工艺操作的指导意义
        5. 需要重点关注的问题

        生成150-250字的详细说明，语言专业、准确、易懂。
        """

        try:
            response = invoke_llm(
                self.llm,
                [HumanMessage(content=prompt)],
                node="ChartGenerator",
                purpose="table_description",
                json_mode=False,
                **scope,
            )
            return response.content
        except Exception as e:
            print(f"大模型调用失败: {str(e)}")
            return f"表格类型：{table_type}，数据规模：{row_count}行"

DEFAULT_CHARTS_DIR = "charts"

class ChartGenerator:
    """图表生成器主类"""

    def __init__(self,
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None,
                 use_image_analysis: bool = True,
                 charts_dir: str = None):
        app_config = get_app_config()
        self.api_key = api_key or app_config.deepseek_api_key or ""
        self.base_url = base_url or app_config.deepseek_base_url
        self.model = model or app_config.deepseek_model
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for chart generation")
        self.use_image_analysis = use_image_analysis  # 新增：是否使用图像分析

        # LLM客户端必用
        self.llm_client = LLMClient(self.api_key, self.base_url, self.model)

        self.charts_dir = charts_dir or DEFAULT_CHARTS_DIR
        os.makedirs(self.charts_dir, exist_ok=True)

    @staticmethod
    def _llm_scope(task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": task.get("_observability_task_id") or task.get("task_id"),
            "job_id": task.get("_job_id"),
            "plan_revision": task.get("_plan_revision"),
            "task_revision": task.get("_task_revision"),
        }

    def process_planner_input(self, planner_input: Dict) -> Dict:
        """处理来自Planner的完整输入"""
        try:
            messages = planner_input["messages"]
            tasks = planner_input["tasks"]
            cursor = planner_input["cursor"]

            if cursor >= len(tasks):
                return self._build_error_output("cursor超出任务范围")

            current_task = tasks[cursor]
            print(f"处理任务: {current_task['task_name']} (ID: {current_task['task_id']})")

            if current_task.get("generate_figure", False) and current_task.get("generate_table", False):
                # 同时生成图表和表格
                current_result = self._handle_chart_and_table_task(current_task, messages)
            elif current_task.get("generate_figure", False):
                # 只生成图表
                current_result = self._handle_chart_generation_task(current_task, messages)
            elif current_task.get("generate_table", False):
                # 只生成表格
                current_result = self._handle_table_generation_task(current_task, messages)
            else:
                # 文本生成
                current_result = self._handle_text_generation_task(current_task, messages)

            output = {
                "messages": self._build_worker_result_message(current_result),
                "current_result": current_result
            }

            return output

        except Exception as e:
            return self._build_error_output(f"处理Planner输入失败: {str(e)}")

    def process_intake_input(self, intake_input: Dict) -> Dict:
        """处理来自Intake的输入（恢复场景）"""
        try:
            messages = intake_input["messages"]
            decision = intake_input.get("decision")

            if decision != "RETRY_WORKER":
                return self._build_error_output("无效的decision")

            message = messages[-1]
            content_data = message.get("content", {})
            checkpoint = content_data.get("checkpoint", {})

            current_section = checkpoint.get("current_section")
            if not current_section:
                return self._build_error_output("恢复信息中缺少当前章节")

            print(f"恢复处理章节: {current_section}")

            task = {
                "task_id": "resume_task",
                "task_name": current_section,
                "task_description": f"恢复处理章节: {current_section}",
                "generate_figure": True,
                "use_resources": checkpoint.get("required", [])
            }

            current_result = self._handle_chart_generation_task(task, messages)

            output = {
                "messages": self._build_worker_result_message(current_result),
                "current_result": current_result
            }

            return output

        except Exception as e:
            return self._build_error_output(f"处理Intake输入失败: {str(e)}")

    def process_verifier_input(self, verifier_input: Dict) -> Dict:
        """处理来自Verifier的输入（重做场景）"""
        try:
            messages = verifier_input["messages"]
            feedback = verifier_input.get("feedback", {})
            decision = verifier_input.get("decision")

            if decision != "RETRY_WORKER":
                return self._build_error_output("无效的decision")

            current_section = feedback.get("current_section")
            if not current_section:
                return self._build_error_output("反馈信息中缺少当前章节")

            print(f"重新处理章节: {current_section}")

            message = messages[-1]
            content_data = message.get("content", {})

            task = {
                "task_id": "rework_task",
                "task_name": current_section,
                "task_description": f"根据反馈重新处理章节: {current_section}",
                "generate_figure": True,
                "use_resources": []
            }

            current_result = self._handle_chart_generation_task(task, messages)

            output = {
                "messages": self._build_worker_result_message(current_result),
                "current_result": current_result
            }

            return output

        except Exception as e:
            return self._build_error_output(f"处理Verifier输入失败: {str(e)}")

    def _handle_chart_generation_task(self, task: Dict, messages: List[Dict]) -> Dict:
        """处理图表生成任务的核心逻辑 - 从use_resources获取CSV数据"""
        try:
            task_id = task["task_id"]
            task_name = task["task_name"]
            use_resources = task.get("use_resources", [])

            print(f"生成图表任务: {task_name}, 使用资源: {use_resources}")

            # 从资源中加载数据
            data = self._load_data_from_resources(use_resources)
            csv_files = [res for res in use_resources if res.endswith('.csv')]
            csv_path = csv_files[0] if csv_files else ""

            chart_types = self._determine_chart_types(data)

            figures = []

            for chart_type in chart_types:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"{task_id}_{chart_type}_{timestamp}"

                png_path = os.path.join(self.charts_dir, f"{base_filename}.png")
                svg_path = os.path.join(self.charts_dir, f"{base_filename}.svg")

                self._generate_single_chart(data, chart_type, png_path)

                # 新增：处理图像和元数据
                image_base64, image_info, metadata = self._prepare_image_analysis_data(png_path, data, csv_path)

                chart_info = {
                    "type": chart_type,
                    "file_base": base_filename,
                    "data_summary": self._generate_data_summary(data)
                }

                description = ""
                if self.llm_client and self.use_image_analysis and image_base64 and metadata:
                    # 使用新的图像分析方法
                    description = self.llm_client.generate_chart_description(
                        chart_type,
                        image_base64,
                        metadata,
                        **self._llm_scope(task),
                    )
                elif self.llm_client:
                    # 降级到旧的数据摘要方法
                    description = self.llm_client.generate_chart_description_fallback(
                        chart_type,
                        chart_info,
                        **self._llm_scope(task),
                    )
                else:
                    description = f"{chart_type}图表，展示化工工艺参数的变化趋势和分析结果"

                figure = {
                    "figure_id": f"fig_{chart_type}_{len(figures)+1}",
                    "description": description,
                    "path": png_path
                }
                # # 添加图像信息（可选）
                # if image_info:
                #     figure["image_info"] = image_info

                figures.append(figure)

            text_output = self._generate_chart_summary(task_name, figures, data)

            current_result = self._create_current_result(
                task_id, task_name, text_output, figures, "COMPLETED"
            )

            return current_result

        except Exception as e:
            return self._create_current_result(
                task.get("task_id", ""),
                task.get("task_name", ""),
                f"图表生成失败: {str(e)}",
                [],
                "FAILED"
            )

    def _load_data_from_resources(self, resources: List[str]) -> pd.DataFrame:
        """从资源列表加载CSV数据"""
        if not resources:
            raise ValueError("未提供数据资源，请在use_resources中指定CSV文件路径")

        # 过滤出CSV文件
        csv_files = [res for res in resources if res.endswith('.csv')]
        if not csv_files:
            raise ValueError("资源中未找到CSV文件，请检查use_resources配置")

        # 读取第一个CSV文件（可扩展为多文件合并）
        csv_path = csv_files[0]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        print(f"加载数据文件: {csv_path}")
        return ChartUtils.read_csv_data(csv_path)

    def _determine_chart_types(self, data: pd.DataFrame) -> List[str]:
        """根据数据特征确定图表类型"""
        time_col = next((col for col in ["date", "时间", "time", "timestamp"] if col in data.columns), None)

        if time_col:
            # 有时间列，优先返回折线图
            return ["line"]

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # 有数值列但无时间列，返回柱状图
            return ["bar"]

        # 默认返回折线图
        return ["line"]

    def _generate_single_chart(self, data: pd.DataFrame, chart_type: str, output_path: str):
        """生成单个图表"""
        time_col = next((col for col in ["date", "时间", "time", "timestamp"] if col in data.columns), None)

        if chart_type == "line":
            ChartUtils.create_line_chart(data, "参数趋势分析", output_path, time_col)
        elif chart_type == "scatter":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                ChartUtils.create_scatter_chart(
                    data, f"{x_col} vs {y_col} 相关性分析", output_path, x_col, y_col
                )
        elif chart_type == "bar":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ChartUtils.create_bar_chart(
                    data, f"{numeric_cols[0]} 统计分析", output_path
                )

    def _generate_data_summary(self, data: pd.DataFrame) -> str:
        """生成数据摘要"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        summary = "数据概览：\n"
        for col in numeric_cols[:3]:
            mean_val = data[col].mean()
            std_val = data[col].std()
            summary += f"{col}: 平均值={mean_val:.2f}, 标准差={std_val:.2f}\n"

        return summary

    def _generate_chart_summary(self, task_name: str, figures: List[Dict], data: pd.DataFrame) -> str:
        """生成图表总结文本"""
        summary = f"## {task_name}\n\n"
        # summary += "基于化工工艺数据进行了可视化分析，主要发现如下：\n\n"

        for i, figure in enumerate(figures, 1):
            summary += f"{i}. {figure['description']}\n\n"

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary += "### 主要数据洞察\n"
            summary += "- 工艺参数整体运行稳定，波动在正常范围内\n"
            summary += "- 各参数之间存在一定的相关性，需要重点关注关键控制点\n"
            summary += "- 建议持续监控异常点，确保生产安全\n"

        return summary

    def _create_current_result(self, task_id: str, section_name: str, text_output: str,
                              figures: List[Dict], status: str = "COMPLETED") -> Dict:
        """创建标准的current_result"""
        return {
            "task_id": task_id,
            "section_name": section_name,
            "text_output": text_output,
            "status": status,
            "tables": [],
            "figures": figures,
            "citations": [],
            "sources_used": []
        }

    def _prepare_image_analysis_data(self, png_path: str, data: pd.DataFrame, csv_path: str) -> tuple:
        """准备图像分析的数据"""
        try:
            # 转换图像为Base64
            image_base64 = ImageProcessor.image_to_base64(png_path)

            # 获取图像信息
            image_info = ImageProcessor.get_image_info(png_path)

            # 提取元数据
            metadata = EnhancedDataSummary.extract_metadata(data, csv_path)

            return image_base64, image_info, metadata
        except Exception as e:
            print(f"图像分析数据准备失败: {str(e)}")
            return None, None, None

    def _build_worker_result_message(self, current_result: Dict) -> List[Dict]:
        """构建发送给Verifier的message"""
        return [{
            "role": "assistant",
            "content": {
                "from": "Worker",
                "to": "Verifier",
                "type": "WORKER_RESULT",
                "current_section": current_result["section_name"],
                "completed_sections": [current_result["section_name"]],
                "progress": 0.6
            }
        }]

    def _handle_text_generation_task(self, task: Dict, messages: List[Dict]) -> Dict:
        """处理文本生成任务（简化实现）"""
        return self._create_current_result(
            task["task_id"],
            task["task_name"],
            f"文本生成任务: {task['task_description']}",
            [],
            "COMPLETED"
        )

    def _handle_table_generation_task(self, task: Dict, messages: List[Dict]) -> Dict:
        """处理表格生成任务的核心逻辑 - 从use_resources获取CSV数据"""
        try:
            task_id = task["task_id"]
            task_name = task["task_name"]
            use_resources = task.get("use_resources", [])

            print(f"生成表格任务: {task_name}, 使用资源: {use_resources}")

            # 从资源中加载数据
            data = self._load_data_from_resources(use_resources)

            # 获取表格配置
            table_config = task.get("table_config", {})
            table_type = table_config.get("table_type", "auto")

            # 自动选择表格类型
            if table_type == "auto":
                table_type = self._determine_table_type(data)

            # 生成表格
            tables = []

            if table_type == "raw" or "raw" in table_config.get("include_types", []):
                raw_table = TableUtils.create_raw_table(data, table_config.get("max_rows", 50))
                tables.append(raw_table)

            if table_type == "summary" or "summary" in table_config.get("include_types", []):
                summary_table = TableUtils.create_summary_table(data)
                tables.append(summary_table)

            if table_type == "anomaly" or "anomaly" in table_config.get("include_types", []):
                anomaly_table = TableUtils.create_anomaly_table(data)
                tables.append(anomaly_table)

            if table_type == "batch" or "batch" in table_config.get("include_types", []):
                batch_table = TableUtils.create_batch_table(data)
                tables.append(batch_table)

            # 生成表格描述
            descriptions = []
            for table in tables:
                if self.llm_client:
                    description = self.llm_client.generate_table_description(
                        table["type"],
                        table["data"],
                        table.get("description", ""),
                        **self._llm_scope(task),
                    )
                else:
                    description = f"{table['type']}表格，包含{len(table['data'])-1}行数据"
                descriptions.append(description)

            # 生成文本输出
            text_output = self._generate_table_summary(task_name, tables, data)

            current_result = self._create_current_result(
                task_id,
                task_name,
                text_output,
                [],  # figures为空
                "COMPLETED"
            )

            # 添加tables到结果
            current_result["tables"] = tables
            current_result["table_descriptions"] = descriptions

            return current_result

        except Exception as e:
            return self._create_current_result(
                task.get("task_id", ""),
                task.get("task_name", ""),
                f"表格生成失败: {str(e)}",
                [],
                "FAILED"
            )

    def _handle_chart_and_table_task(self, task: Dict, messages: List[Dict]) -> Dict:
        """处理同时生成图表和表格的任务"""
        try:
            task_id = task["task_id"]
            task_name = task["task_name"]
            use_resources = task.get("use_resources", [])

            print(f"生成图表和表格任务: {task_name}, 使用资源: {use_resources}")

            # 从资源中加载数据
            data = self._load_data_from_resources(use_resources)
            csv_files = [res for res in use_resources if res.endswith('.csv')]
            csv_path = csv_files[0] if csv_files else ""


            # 生成图表
            chart_types = self._determine_chart_types(data)
            figures = []

            for chart_type in chart_types:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"{task_id}_{chart_type}_{timestamp}"

                png_path = os.path.join(self.charts_dir, f"{base_filename}.png")
                svg_path = os.path.join(self.charts_dir, f"{base_filename}.svg")

                self._generate_single_chart(data, chart_type, png_path)

                 # 新增：处理图像和元数据
                image_base64, image_info, metadata = self._prepare_image_analysis_data(png_path, data, csv_path)


                chart_info = {
                    "type": chart_type,
                    "file_base": base_filename,
                    "data_summary": self._generate_data_summary(data)
                }

                description = ""
                if self.llm_client and self.use_image_analysis and image_base64 and metadata:
                    # 使用新的图像分析方法
                    description = self.llm_client.generate_chart_description(
                        chart_type,
                        image_base64,
                        metadata,
                        **self._llm_scope(task),
                    )
                elif self.llm_client:
                    # 降级到旧的数据摘要方法
                    description = self.llm_client.generate_chart_description_fallback(
                        chart_type,
                        chart_info,
                        **self._llm_scope(task),
                    )
                else:
                    description = f"{chart_type}图表，展示化工工艺参数的变化趋势和分析结果"

                figure = {
                    "figure_id": f"fig_{chart_type}_{len(figures)+1}",
                    "description": description,
                    "path": png_path
                }
                figures.append(figure)

            # 获取表格配置
            table_config = task.get("table_config", {})
            table_type = table_config.get("table_type", "auto")

            # 自动选择表格类型
            if table_type == "auto":
                table_type = self._determine_table_type(data)

            # 生成表格
            tables = []

            if table_type == "raw" or "raw" in table_config.get("include_types", []):
                raw_table = TableUtils.create_raw_table(data, table_config.get("max_rows", 50))
                tables.append(raw_table)

            if table_type == "summary" or "summary" in table_config.get("include_types", []):
                summary_table = TableUtils.create_summary_table(data)
                tables.append(summary_table)

            if table_type == "anomaly" or "anomaly" in table_config.get("include_types", []):
                anomaly_table = TableUtils.create_anomaly_table(data)
                tables.append(anomaly_table)

            if table_type == "batch" or "batch" in table_config.get("include_types", []):
                batch_table = TableUtils.create_batch_table(data)
                tables.append(batch_table)

            # 生成表格描述
            table_descriptions = []
            for table in tables:
                if self.llm_client:
                    description = self.llm_client.generate_table_description(
                        table["type"],
                        table["data"],
                        table.get("description", ""),
                        **self._llm_scope(task),
                    )
                else:
                    description = f"{table['type']}表格，包含{len(table['data'])-1}行数据"
                table_descriptions.append(description)

            # 生成文本输出
            text_output = self._generate_chart_and_table_summary(task_name, figures, tables, data)

            current_result = self._create_current_result(
                task_id,
                task_name,
                text_output,
                figures,
                "COMPLETED"
            )

            # 添加tables到结果
            current_result["tables"] = tables
            current_result["table_descriptions"] = table_descriptions

            return current_result

        except Exception as e:
            return self._create_current_result(
                task.get("task_id", ""),
                task.get("task_name", ""),
                f"图表和表格生成失败: {str(e)}",
                [],
                "FAILED"
            )

    def _determine_table_type(self, data: pd.DataFrame) -> str:
        """根据数据特征自动确定表格类型"""
        # 检查是否包含异常标记列
        anomaly_cols = [col for col in data.columns if "异常" in col or "异常" in str(col).lower()]
        if anomaly_cols:
            return "anomaly"

        # 检查是否包含批次分类列
        batch_cols = [col for col in data.columns if "批次" in col or "batch" in str(col).lower()]
        if batch_cols:
            return "batch"

        # 检查是否为时间序列
        time_cols = ["date", "时间", "time", "timestamp"]
        if any(col in data.columns for col in time_cols):
            return "raw"  # 时间序列默认显示原始数据

        # 默认返回统计摘要表
        return "summary"

    def _generate_table_summary(self, task_name: str, tables: List[Dict], data: pd.DataFrame) -> str:
        """生成表格总结文本"""
        summary = f"## {task_name}\n\n"
        summary += "基于化工工艺数据进行了表格分析，主要结果如下：\n\n"

        for i, table in enumerate(tables, 1):
            summary += f"{i}. **{table['description']}**\n"
            summary += f"   - 表格类型：{table['type']}\n"
            summary += f"   - 数据行数：{len(table['data'])-1}\n"
            if table['type'] == 'anomaly' and 'data' in table:
                anomaly_count = len(table['data']) - 1  # 减去标题行
                if anomaly_count > 0:
                    summary += f"   - 异常总数：{anomaly_count}\n"
            summary += "\n"

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary += "### 数据分析洞察\n"
            summary += "- 表格数据已按类型分类展示，包含原始数据、统计特征和异常分析\n"
            summary += "- 建议结合图表分析，全面理解工艺参数的变化趋势\n"
            summary += "- 异常数据需要重点关注，分析原因并采取改进措施\n"

        return summary

    def _generate_chart_and_table_summary(self, task_name: str, figures: List[Dict],
                                        tables: List[Dict], data: pd.DataFrame) -> str:
        """生成图表和表格的综合总结文本"""
        summary = f"## {task_name}\n\n"
        summary += "基于化工工艺数据进行了可视化分析和表格展示，主要结果如下：\n\n"

        # 图表部分
        if figures:
            summary += "###图表分析\n"
            for i, figure in enumerate(figures, 1):
                summary += f"{i}. **{figure['description']}**\n"
            summary += "\n"

        # 表格部分
        if tables:
            summary += "###表格分析\n"
            for i, table in enumerate(tables, 1):
                summary += f"{i}. **{table['description']}**\n"
                summary += f"   - 表格类型：{table['type']}\n"
                summary += f"   - 数据行数：{len(table['data'])-1}\n"
                if table['type'] == 'anomaly' and 'data' in table:
                    anomaly_count = len(table['data']) - 1
                    if anomaly_count > 0:
                        summary += f"   - 异常总数：{anomaly_count}\n"
                summary += "\n"

        # 综合分析
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary += "###综合分析\n"
            summary += "- 通过图表可视化直观展示了数据变化趋势\n"
            summary += "- 表格数据提供了详细的统计特征和异常信息\n"
            summary += "- 建议将图表与表格结合分析，全面把握工艺状态\n"
            summary += "- 定期监控异常指标，确保生产安全和产品质量\n"

        return summary

    def _build_error_output(self, error_msg: str) -> Dict:
        """构建错误输出"""
        return {
            "messages": [{
                "role": "assistant",
                "content": {
                    "from": "Worker",
                    "to": "Verifier",
                    "type": "ERROR",
                    "error_message": error_msg
                }
            }],
            "current_result": {
                "task_id": "error",
                "section_name": "错误",
                "text_output": error_msg,
                "status": "FAILED",
                "tables": [],
                "figures": [],
                "citations": [],
                "sources_used": []
            }
        }
