"""
CSV Worker 系统入口文件
负责处理 JSON 格式的输入输出，与系统其他模块对接
"""
import json
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

class CSVAnalysisTool:
    """CSV数据分析工具"""

    def __init__(self):
        self.generate_charts = False  # 禁用图表生成

    def analyze_csv_file(self, file_path: str, analysis_type: str = "basic") -> Dict:
        """分析单个CSV文件"""
        try:
            if not os.path.exists(file_path):
                return self._generate_error_result(f"文件不存在: {file_path}")

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 基本分析
            basic_stats = self._get_basic_statistics(df)

            # 根据分析类型进行不同处理
            if analysis_type == "technical":
                analysis_result = self._technical_analysis(df, file_path)
            elif analysis_type == "statistical":
                analysis_result = self._statistical_analysis(df, file_path)
            else:  # basic
                analysis_result = self._basic_analysis(df, file_path)

            return {
                "success": True,
                "file_path": file_path,
                "basic_statistics": basic_stats,
                "analysis_type": analysis_type,
                "results": analysis_result,
                "charts": []  # 空列表，不生成图表
            }

        except Exception as e:
            return self._generate_error_result(f"分析CSV文件失败: {str(e)}")

    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """获取基本统计信息"""
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }

    def _basic_analysis(self, df: pd.DataFrame, file_path: str) -> Dict:
        """基础分析"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        analysis = {
            "summary": f"对文件 {os.path.basename(file_path)} 进行了基础分析",
            "numeric_columns": numeric_cols.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
        }

        if len(numeric_cols) > 0:
            analysis["descriptive_stats"] = df[numeric_cols].describe().to_dict()
            analysis["correlation_matrix"] = df[numeric_cols].corr().to_dict()

        return analysis

    def _technical_analysis(self, df: pd.DataFrame, file_path: str) -> Dict:
        """技术分析（针对化工工艺数据优化）"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        analysis = {
            "summary": f"对化工工艺数据文件 {os.path.basename(file_path)} 进行了技术分析",
            "process_parameters": {},
            "quality_metrics": {},
            "efficiency_indicators": {}
        }

        # 识别常见的化工工艺参数
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['温度', 'temp', 'pressure', '压力']):
                analysis["process_parameters"][col] = self._analyze_process_parameter(df[col])
            elif any(keyword in col_lower for keyword in ['纯度', '质量', 'quality', 'purity']):
                analysis["quality_metrics"][col] = self._analyze_quality_metric(df[col])
            elif any(keyword in col_lower for keyword in ['效率', '转化率', 'efficiency', 'conversion']):
                analysis["efficiency_indicators"][col] = self._analyze_efficiency_indicator(df[col])

        # 时间序列分析（如果存在时间列）
        time_cols = [col for col in df.columns if 'time' in col.lower() or '时间' in col or 'date' in col]
        if time_cols:
            analysis["time_series_analysis"] = self._time_series_analysis(df, time_cols[0])

        return analysis

    def _statistical_analysis(self, df: pd.DataFrame, file_path: str) -> Dict:
        """统计分析"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        analysis = {
            "summary": f"对文件 {os.path.basename(file_path)} 进行了统计分析",
            "hypothesis_testing": {},
            "anova_analysis": {},
            "regression_analysis": {}
        }

        if len(numeric_cols) >= 2:
            # 简单的相关性分析
            analysis["correlation_analysis"] = {
                "pearson_correlation": df[numeric_cols].corr(method='pearson').to_dict(),
                "spearman_correlation": df[numeric_cols].corr(method='spearman').to_dict()
            }

        return analysis

    def _analyze_process_parameter(self, series: pd.Series) -> Dict:
        """分析工艺参数"""
        return {
            "control_limits": {
                "upper": series.mean() + 2 * series.std(),
                "lower": series.mean() - 2 * series.std(),
                "mean": series.mean()
            },
            "stability": "stable" if series.std() / series.mean() < 0.1 else "unstable",
            "trend": "increasing" if series.iloc[-1] > series.iloc[0] else "decreasing" if series.iloc[-1] <
                                                                                           series.iloc[0] else "stable"
        }

    def _analyze_quality_metric(self, series: pd.Series) -> Dict:
        """分析质量指标"""
        return {
            "specification_limits": {
                "target": series.mean(),
                "tolerance": series.std()
            },
            "cpk": abs((series.mean() - series.quantile(0.001)) / (3 * series.std())) if series.std() > 0 else 0,
            "yield": len(series[series > series.quantile(0.05)]) / len(series) if len(series) > 0 else 0
        }

    def _analyze_efficiency_indicator(self, series: pd.Series) -> Dict:
        """分析效率指标"""
        return {
            "performance": "excellent" if series.mean() > series.quantile(
                0.8) else "good" if series.mean() > series.quantile(0.6) else "poor",
            "improvement_potential": series.max() - series.mean(),
            "consistency": "consistent" if series.std() / series.mean() < 0.05 else "variable"
        }

    def _time_series_analysis(self, df: pd.DataFrame, time_col: str) -> Dict:
        """时间序列分析"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        analysis = {}
        for col in numeric_cols:
            if col != time_col:
                analysis[col] = {
                    "trend": "increasing" if df[col].iloc[-1] > df[col].iloc[0] else "decreasing",
                    "seasonality": "detected" if self._check_seasonality(df[col]) else "not_detected",
                    "volatility": df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                }

        return analysis

    def _check_seasonality(self, series: pd.Series) -> bool:
        """简单检查季节性（简化实现）"""
        if len(series) < 12:
            return False
        # 简单的季节性检测逻辑
        return series.std() > 0.1 * series.mean()

    def _generate_error_result(self, error_msg: str) -> Dict:
        """生成错误结果"""
        return {
            "success": False,
            "error": error_msg,
            "file_path": "",
            "basic_statistics": {},
            "analysis_type": "",
            "results": {},
            "charts": []
        }


class CSVWorkerSystem:
    """CSV Worker 系统适配器"""

    def __init__(self):
        self.analysis_tool = CSVAnalysisTool()

    def process_system_input(self, input_data: Dict) -> Dict:
        """处理系统输入，生成系统输出"""
        try:
            print("📊 开始处理CSV分析任务...")

            # 1. 解析系统输入
            task_spec = input_data.get('task_spec', {})
            resources = input_data.get('resources', [])

            # 2. 提取CSV文件路径
            csv_files = []
            for resource in resources:
                if isinstance(resource, dict) and 'path' in resource:
                    path = resource['path']
                    if path.endswith(('.csv', '.xlsx', '.xls')):
                        csv_files.append(path)
                elif isinstance(resource, str) and resource.endswith(('.csv', '.xlsx', '.xls')):
                    csv_files.append(resource)

            if not csv_files:
                return self._generate_error_output(input_data, "未找到CSV或Excel文件资源")

            print(f"找到 {len(csv_files)} 个数据文件: {csv_files}")

            # 3. 分析每个CSV文件
            analysis_results = []
            for csv_file in csv_files:
                print(f"分析文件: {csv_file}")
                result = self.analysis_tool.analyze_csv_file(
                    csv_file,
                    task_spec.get('analysis_type', 'technical')
                )
                analysis_results.append(result)

            # 4. 生成综合报告
            combined_result = self._generate_combined_report(analysis_results)

            # 5. 转换为系统输出格式
            system_output = self._convert_to_system_output(input_data, combined_result)

            print("✅ CSV分析任务完成")
            return system_output

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ CSV分析失败: {str(e)}")
            print(f"错误详情: {error_details}")
            return self._generate_error_output(input_data, f"CSV分析失败: {str(e)}")

    def _generate_combined_report(self, analysis_results: List[Dict]) -> Dict:
        """生成综合分析报告"""
        successful_analyses = [r for r in analysis_results if r.get('success', False)]
        failed_analyses = [r for r in analysis_results if not r.get('success', False)]

        return {
            "success": len(successful_analyses) > 0,
            "total_files": len(analysis_results),
            "successful_files": len(successful_analyses),
            "failed_files": len(failed_analyses),
            "analysis_results": analysis_results,
            "summary": self._generate_summary_statistics(successful_analyses),
            "charts": [],  # 空列表，不生成图表
            "recommendations": self._generate_recommendations(successful_analyses)
        }

    def _generate_summary_statistics(self, results: List[Dict]) -> Dict:
        """生成汇总统计"""
        if not results:
            return {}

        total_rows = sum(r['basic_statistics'].get('row_count', 0) for r in results)
        total_columns = sum(r['basic_statistics'].get('column_count', 0) for r in results)

        return {
            "total_data_points": total_rows,
            "total_variables": total_columns,
            "average_rows_per_file": total_rows / len(results) if results else 0,
            "file_count": len(results)
        }

    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """生成分析建议"""
        recommendations = []

        for result in results:
            stats = result.get('basic_statistics', {})
            missing_values = stats.get('missing_values', {})

            # 检查缺失值
            total_missing = sum(missing_values.values())
            if total_missing > 0:
                recommendations.append(f"文件 {result['file_path']} 存在 {total_missing} 个缺失值，建议进行数据清洗")

            # 检查数据质量
            if stats.get('row_count', 0) < 10:
                recommendations.append(f"文件 {result['file_path']} 数据量较少，建议收集更多数据")

        if not recommendations:
            recommendations.append("数据质量良好，可以进行深入的工艺分析和优化")

        return recommendations

    def _convert_to_system_output(self, input_data: Dict, analysis_result: Dict) -> Dict:
        """将分析结果转换为系统输出格式"""

        # 构建文本输出
        text_output = self._build_text_output(analysis_result)

        # 构建表格信息
        tables = []
        for i, result in enumerate(analysis_result.get('analysis_results', [])):
            if result.get('success'):
                tables.append({
                    "table_id": f"csv_table_{i + 1}",
                    "description": f"{os.path.basename(result['file_path'])} 数据概览",
                    "data": self._convert_stats_to_table(result['basic_statistics'])
                })

        return {
            "request_id": input_data.get('request_id', ''),
            "session_id": input_data.get('session_id', ''),
            "decision": "NEXT",
            "messages": [
                {
                    "role": "system",
                    "content": "WORKER_COMPLETE"
                },
                {
                    "role": "assistant",
                    "content": {
                        "type": "CSV_ANALYSIS_RESULT",
                        "analysis_result": analysis_result
                    }
                }
            ],
            "current_result": {
                "task_id": input_data.get('request_id', 'csv_analysis'),
                "section_name": "CSV数据分析",
                "text_output": text_output,
                "status": "COMPLETED",
                "tables": tables,
                "figures": [],  # 空列表，不生成图表
                "citations": [],
                "sources_used": input_data.get('resources', [])
            },
            "resources": [],
            "checkpoint": None,
            "new_resources": [],
            "new_content": [],
            "task_spec": None,
            "resume_from_step": None,
            "requested_by": "csv_worker"
        }

    def _build_text_output(self, analysis_result: Dict) -> str:
        """构建文本输出"""
        text = "# CSV数据分析报告\n\n"

        if not analysis_result.get('success'):
            text += "## 分析状态\n分析过程中遇到问题，请检查输入文件。\n\n"
            return text

        summary = analysis_result.get('summary', {})
        text += f"## 分析概览\n"
        text += f"- 分析文件数量: {summary.get('file_count', 0)}\n"
        text += f"- 总数据点数: {summary.get('total_data_points', 0)}\n"
        text += f"- 总变量数: {summary.get('total_variables', 0)}\n\n"

        text += "## 详细分析结果\n"
        for result in analysis_result.get('analysis_results', []):
            if result.get('success'):
                file_name = os.path.basename(result['file_path'])
                stats = result['basic_statistics']
                text += f"### {file_name}\n"
                text += f"- 行数: {stats.get('row_count', 0)}\n"
                text += f"- 列数: {stats.get('column_count', 0)}\n"
                text += f"- 数据列: {', '.join(stats.get('columns', []))}\n\n"

        text += "## 分析建议\n"
        for recommendation in analysis_result.get('recommendations', []):
            text += f"- {recommendation}\n"

        return text

    def _convert_stats_to_table(self, stats: Dict) -> List[List]:
        """将统计信息转换为表格格式"""
        table = [["统计项", "数值"]]

        table.append(["数据行数", str(stats.get('row_count', 0))])
        table.append(["数据列数", str(stats.get('column_count', 0))])
        table.append(["内存使用", f"{stats.get('memory_usage', 0)} bytes"])

        return table

    def _generate_error_output(self, input_data: Dict, error_msg: str) -> Dict:
        """生成错误输出"""
        return {
            "request_id": input_data.get('request_id', ''),
            "session_id": input_data.get('session_id', ''),
            "decision": "RETRY_WORKER",
            "messages": [
                {
                    "role": "system",
                    "content": "WORKER_ERROR"
                },
                {
                    "role": "assistant",
                    "content": {
                        "type": "CSV_ANALYSIS_ERROR",
                        "error_info": {
                            "error_type": "processing_error",
                            "error_message": error_msg
                        }
                    }
                }
            ],
            "current_result": {
                "task_id": input_data.get('request_id', 'csv_analysis'),
                "section_name": "CSV数据分析",
                "text_output": f"CSV分析失败: {error_msg}",
                "status": "FAILED",
                "tables": [],
                "figures": [],  # 空列表
                "citations": [],
                "sources_used": []
            },
            "checkpoint": {
                "node": "csv_worker",
                "stage": "processing",
                "checkpoint_type": "system_error"
            },
            "requested_by": "csv_worker"
        }


def main():
    """主函数 - 从标准输入读取，输出到标准输出"""
    try:
        # 读取输入
        input_json = sys.stdin.read().strip()
        if not input_json:
            raise ValueError("输入为空")

        input_data = json.loads(input_json)

        # 处理
        worker_system = CSVWorkerSystem()
        output_data = worker_system.process_system_input(input_data)

        # 输出
        print(json.dumps(output_data, ensure_ascii=False, indent=2))

    except json.JSONDecodeError as e:
        error_output = {
            "decision": "RETRY_WORKER",
            "messages": [{
                "role": "system",
                "content": "WORKER_ERROR"
            }],
            "error": f"JSON解析失败: {str(e)}"
        }
        print(json.dumps(error_output, ensure_ascii=False, indent=2))
    except Exception as e:
        error_output = {
            "decision": "RETRY_WORKER",
            "messages": [{
                "role": "system",
                "content": "WORKER_ERROR"
            }],
            "error": f"系统错误: {str(e)}"
        }
        print(json.dumps(error_output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()