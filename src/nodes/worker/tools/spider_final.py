import requests
import json
import re
import time
import random
import sys
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
import urllib3
from bs4 import BeautifulSoup
import logging
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse
import jieba
import jieba.analyse
import jieba.posseg as pseg
from collections import Counter

# 针对 Windows 平台的异步子进程支持修复
if sys.platform == 'win32':
    try:
        # 设置 ProactorEventLoopPolicy 以支持 subprocess
        if isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception as e:
        print(f"设置 WindowsProactorEventLoopPolicy 失败: {e}")

# 关闭urllib3的InsecureRequestWarning警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - Worker-网页抓取 - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler("worker_scrape_log.log", encoding="utf-8"),
#         logging.StreamHandler()
#     ]
# )

# 初始化jieba分词
jieba.initialize()


class ChemicalKeywordExtractor:
    """化工领域关键词提取器"""

    def __init__(self):
        # 化工领域停用词（通用词汇）
        self.stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
            '说',
            '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '什么', '我们', '吗', '可以', '这个',
            '这样',
            '对于', '关于', '以及', '或者', '例如', '比如', '如下', '包括', '具有', '进行', '通过', '使用', '需要',
            '主要',
            '重要', '相关', '各种', '不同', '相同', '类似', '特别', '尤其', '非常', '十分', '更加', '比较', '相对',
            '一定',
            '一些', '一点', '一种', '方面', '部分', '整体', '全部', '完全', '基本', '根本', '主要', '次要', '其他',
            '其余',
            '另外', '此外', '同时', '然后', '最后', '总之', '因此', '所以', '因为', '由于', '为了', '使得', '导致',
            '造成',
            '成为', '作为', '变成', '形成', '产生', '发生', '出现', '存在', '属于', '位于', '处于', '涉及', '包含',
            '涵盖',
            '研究', '分析', '探讨', '讨论', '介绍', '说明', '解释', '阐述', '描述', '总结', '归纳', '概括', '评估',
            '评价',
            '测试', '实验', '试验', '验证', '确认', '确定', '证明', '显示', '表明', '揭示', '发现', '提出', '建议',
            '推荐',
            '方法', '方式', '手段', '途径', '策略', '技术', '工艺', '过程', '流程', '步骤', '阶段', '环节', '系统',
            '体系',
            '结构', '组成', '成分', '含量', '浓度', '比例', '比率', '参数', '指标', '标准', '规范', '要求', '条件',
            '环境',
            '温度', '压力', '时间', '速度', '效率', '效果', '性能', '特性', '特点', '优势', '缺点', '问题', '挑战',
            '困难',
            '发展', '进展', '趋势', '方向', '前景', '未来', '现状', '历史', '背景', '意义', '价值', '作用', '功能',
            '用途',
            '应用', '使用', '利用', '处理', '加工', '生产', '制造', '制备', '合成', '反应', '转化', '变化', '改变',
            '调整',
            '优化', '改进', '改善', '提高', '提升', '增强', '降低', '减少', '控制', '调节', '管理', '维护', '保养',
            '操作',
            '运行', '启动', '停止', '关闭', '打开', '设置', '配置', '安装', '拆卸', '清洗', '清洁', '消毒', '灭菌',
            '检验',
            '检测', '测量', '计量', '计算', '统计', '记录', '报告', '文档', '资料', '信息', '数据', '结果', '结论',
            '成果',
            '产品', '商品', '货物', '材料', '原料', '试剂', '药品', '化学品', '化合物', '混合物', '溶液', '溶剂',
            '溶质',
            '固体', '液体', '气体', '粉末', '颗粒', '晶体', '薄膜', '涂层', '表面', '界面', '内部', '外部', '整体', '局部'
        ])

        # 化工领域专有名词词典（可根据需要扩展）
        self.chemical_terms = set([
            # 化工过程
            '蒸馏', '萃取', '结晶', '过滤', '离心', '干燥', '蒸发', '冷凝', '吸附', '脱附', '催化', '裂解', '聚合',
            '缩合', '水解', '氧化', '还原', '加氢', '脱氢', '磺化', '硝化', '卤化', '烷基化', '酰基化', '酯化',

            # 化工设备
            '反应器', '反应釜', '塔器', '换热器', '冷凝器', '蒸发器', '干燥器', '过滤器', '离心机', '泵', '阀门',
            '管道', '储罐', '槽罐', '搅拌器', '加热器', '冷却器', '分离器', '吸收塔', '精馏塔', '萃取塔',

            # 化工原料和产品
            '石油', '天然气', '煤炭', '原油', '汽油', '柴油', '煤油', '润滑油', '石蜡', '沥青', '乙烯', '丙烯',
            '苯', '甲苯', '二甲苯', '甲醇', '乙醇', '丙醇', '丁醇', '甲醛', '乙醛', '丙酮', '醋酸', '硫酸', '盐酸',
            '硝酸', '磷酸', '烧碱', '纯碱', '小苏打', '化肥', '农药', '塑料', '橡胶', '纤维', '涂料', '染料',
            '颜料', '胶粘剂', '表面活性剂', '催化剂', '助剂', '添加剂', '溶剂', '树脂', '聚合物', '复合材料',

            # 化工技术
            '石油化工', '煤化工', '天然气化工', '精细化工', '生物化工', '医药化工', '食品化工', '环境化工',
            '化工工艺', '化工设计', '化工安全', '化工环保', '化工自动化', '化工仪表', '化工过程控制',

            # 化工指标
            '收率', '转化率', '选择性', '纯度', '浓度', 'pH值', '粘度', '密度', '沸点', '熔点', '闪点', '燃点',
            '爆炸极限', '毒性', '腐蚀性', '稳定性', '活性', '选择性', '寿命', '效率'
        ])

        # 添加化工术语到jieba词典，提高分词准确性
        for term in self.chemical_terms:
            jieba.add_word(term, freq=1000, tag='n')

    def extract_chemical_keywords(self, text: str, num_keywords: int = 3) -> List[str]:
        """
        从化工相关文本中提取关键词

        Args:
            text: 输入文本
            num_keywords: 需要提取的关键词数量

        Returns:
            化工相关关键词列表
        """
        if not text or len(text.strip()) < 10:
            return []

        # 方法1: 使用词性标注提取名词性短语（化工领域多为名词）
        words = pseg.cut(text)
        noun_phrases = []
        current_phrase = []

        for word, flag in words:
            # 保留名词、专有名词、化工术语
            if flag.startswith('n') or word in self.chemical_terms:
                if word not in self.stopwords and len(word) > 1:
                    current_phrase.append(word)
            elif current_phrase:
                # 合并连续的名词组成名词短语
                if len(current_phrase) > 1:
                    noun_phrases.append(''.join(current_phrase))
                elif current_phrase[0] not in self.stopwords:
                    noun_phrases.append(current_phrase[0])
                current_phrase = []

        # 处理最后一个短语
        if current_phrase:
            if len(current_phrase) > 1:
                noun_phrases.append(''.join(current_phrase))
            elif current_phrase[0] not in self.stopwords:
                noun_phrases.append(current_phrase[0])

        # 方法2: 使用TF-IDF提取关键词，但只保留化工相关词
        try:
            keywords_tfidf = jieba.analyse.extract_tags(
                text,
                topK=num_keywords * 3,
                withWeight=False,
                allowPOS=('n', 'ns', 'nr', 'nt', 'nz', 'eng')  # 名词和英文
            )
            # 过滤非化工词汇
            keywords_tfidf = [
                kw for kw in keywords_tfidf
                if kw not in self.stopwords
                   and len(kw) > 1
                   and (kw in self.chemical_terms or self._is_chemical_like(kw))
            ]
        except:
            keywords_tfidf = []

        # 合并两种方法的结果
        all_keywords = list(set(noun_phrases + keywords_tfidf))

        # 按化工相关性排序
        sorted_keywords = sorted(
            all_keywords,
            key=lambda x: (
                2 if x in self.chemical_terms else  # 化工专有词优先
                1 if self._is_chemical_like(x) else  # 化工类词汇次之
                0  # 其他
            ),
            reverse=True
        )

        # 返回指定数量的关键词
        return sorted_keywords[:num_keywords]

    def _is_chemical_like(self, word: str) -> bool:
        """判断词汇是否像化工词汇"""
        # 包含化学元素或化工常见字
        chemical_chars = ['化', '工', '酸', '碱', '盐', '醇', '醛', '酮', '酯', '烃', '烯', '烷', '苯', '胺', '醚',
                          '酚']

        # 检查是否包含化工相关字
        if any(char in word for char in chemical_chars):
            return True

        # 检查是否是化学品命名模式（数字+字母，如C6H6，或包含常见化学符号）
        if re.search(r'[A-Za-z]+\d+', word):  # 如H2O, CH4
            return True

        # 检查是否包含常见化工后缀
        chemical_suffixes = ['化', '剂', '油', '气', '煤', '矿', '石', '膏', '粉', '晶', '胶', '漆', '料', '材']
        if any(word.endswith(suffix) for suffix in chemical_suffixes):
            return True

        return False

    def generate_chemical_search_queries(self, task_description: str, num_queries: int = 2) -> List[str]:
        """
        为化工领域任务生成搜索查询

        Args:
            task_description: 任务描述
            num_queries: 生成的查询数量

        Returns:
            搜索查询列表
        """
        # 提取化工关键词
        keywords = self.extract_chemical_keywords(task_description, num_keywords=min(5, num_queries * 2))

        if not keywords:
            # 如果提取失败，尝试从任务描述中提取核心部分
            sentences = re.split(r'[。！？；]', task_description)
            if sentences and len(sentences[0]) > 5:
                return [sentences[0].strip()[:40]]
            else:
                return [task_description.strip()[:40]]

        # 生成搜索查询（化工领域通常需要具体查询）
        search_queries = []

        # 1. 单关键词查询（化工领域通常需要具体）
        for keyword in keywords[:num_queries]:
            search_queries.append(keyword)

        # 2. 如果关键词是化学品名称，可以加上"制备"、"合成"、"生产"等词
        for keyword in keywords[:min(2, len(keywords))]:
            if self._is_chemical_like(keyword) and keyword not in search_queries:
                # 添加常见的化工动作词
                actions = ['制备', '合成', '生产', '工艺', '技术', '方法', '应用', '用途']
                for action in actions:
                    if len(search_queries) < num_queries:
                        search_queries.append(f"{keyword} {action}")

        # 3. 如果还有空位，添加组合查询
        if len(keywords) >= 2 and len(search_queries) < num_queries:
            search_queries.append(f"{keywords[0]} {keywords[1]}")

        # 去重并确保不超过指定数量
        unique_queries = []
        for query in search_queries:
            if query not in unique_queries and self._is_query_meaningful(query):
                unique_queries.append(query)

        return unique_queries[:num_queries]

    def _is_query_meaningful(self, query: str) -> bool:
        """检查化工查询是否有效"""
        if not query or len(query.strip()) < 2:
            return False

        # 检查是否包含有效字符
        if not re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', query):
            return False

        # 化工查询通常比一般查询长一些
        if len(query.strip()) < 3:
            return False

        return True


DEFAULT_OUTPUT_DIR = "worker_scrape_results"

class WorkerScraper:
    """网页抓取Worker实现类，化工领域专用"""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        # 初始化请求头
        self.default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
            "Referer": "https://cn.bing.com/"
        }

        # 初始化化工关键词提取器
        self.keyword_extractor = ChemicalKeywordExtractor()

        # 搜索缓存
        self.search_cache = {}

    def _save_json_result(self, data: Dict[str, Any], file_path: str) -> None:
        """保存JSON结果到文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"JSON结果已保存：{file_path}")

    def _filter_non_text_content(self, content: str) -> str:
        """过滤内容中的非文字信息"""
        if not content:
            return ""

        # 移除Markdown图片语法
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        # 移除HTML图片标签
        content = re.sub(r'<img[^>]*?>', '', content, flags=re.IGNORECASE)
        # 移除HTML链接标签及内容
        content = re.sub(r'<a\s+[^>]*?>.*?</a>', '', content, flags=re.IGNORECASE | re.DOTALL)
        # 移除URL链接
        content = re.sub(r'https?://\S+', '', content)
        content = re.sub(r'www\.\S+', '', content)
        # 移除图片描述文字
        content = re.sub(r'图\s*\d+[:：]?', '', content)
        content = re.sub(r'示意图[:：]?', '', content)
        # 清理空行和空格
        content = re.sub(r'\n+', '\n', content).strip()
        content = re.sub(r' {2,}', ' ', content)
        # 移除特殊符号序列
        content = re.sub(r'[*#]+', '', content)

        return content

    def extract_chemical_keywords_from_task(self, task_description: str) -> Dict[str, Any]:
        """
        从化工任务描述中提取关键词信息

        Args:
            task_description: 任务描述文本

        Returns:
            包含关键词信息的字典
        """
        logging.info(f"从化工任务描述提取关键词：{task_description[:100]}...")

        # 提取化工关键词
        keywords = self.keyword_extractor.extract_chemical_keywords(task_description, num_keywords=3)

        # 生成化工搜索查询
        search_queries = self.keyword_extractor.generate_chemical_search_queries(task_description, num_queries=2)

        # 分析任务类型
        task_type = self._classify_chemical_task(task_description)

        result = {
            "task_description": task_description,
            "task_type": task_type,
            "extracted_keywords": keywords,
            "search_queries": search_queries,
            "has_meaningful_queries": len(search_queries) > 0
        }

        logging.info(f"化工关键词提取结果：关键词={keywords}, 查询={search_queries}, 类型={task_type}")
        return result

    def _classify_chemical_task(self, description: str) -> str:
        """分类化工任务类型"""
        description_lower = description.lower()

        if any(word in description_lower for word in ['制备', '合成', '生产', '制造']):
            return "制备合成类"
        elif any(word in description_lower for word in ['工艺', '流程', '过程', '方法']):
            return "工艺技术类"
        elif any(word in description_lower for word in ['性质', '性能', '特性', '参数']):
            return "性质参数类"
        elif any(word in description_lower for word in ['应用', '用途', '使用']):
            return "应用用途类"
        elif any(word in description_lower for word in ['市场', '价格', '供需', '趋势']):
            return "市场分析类"
        elif any(word in description_lower for word in ['安全', '环保', '危险', '毒性']):
            return "安全环保类"
        else:
            return "一般查询类"

    def search_chemical_content(self, task_description: str, num_results: int = 5) -> Dict[str, Any]:
        """
        针对化工内容进行搜索

        Args:
            task_description: 任务描述
            num_results: 每个查询返回的结果数量

        Returns:
            搜索结果字典
        """
        # 提取化工关键词和查询
        keyword_info = self.extract_chemical_keywords_from_task(task_description)

        if not keyword_info["has_meaningful_queries"]:
            return {
                "status": "FAILED",
                "message": "无法从任务描述中提取有效化工搜索关键词",
                "keyword_info": keyword_info,
                "search_results": []
            }

        all_results = []
        processed_urls = set()

        # 对每个化工查询进行搜索
        for query in keyword_info["search_queries"]:
            logging.info(f"使用化工查询进行搜索：{query}")

            # 检查缓存
            if query in self.search_cache:
                cached_results = self.search_cache[query]
                for result in cached_results:
                    if result["url"] not in processed_urls:
                        all_results.append(result)
                        processed_urls.add(result["url"])
            else:
                # 执行搜索（化工领域可能需要更具体的搜索）
                query_results = self.search_bing_with_chemical_filter(query, num_results)
                self.search_cache[query] = query_results

                for result in query_results:
                    if result["url"] not in processed_urls:
                        all_results.append(result)
                        processed_urls.add(result["url"])

            # 如果已经收集足够的结果，提前停止
            if len(all_results) >= num_results * 2:
                break

        # 按化工相关性排序
        sorted_results = sorted(
            all_results,
            key=lambda x: self._calculate_chemical_relevance(x, keyword_info["extracted_keywords"]),
            reverse=True
        )

        # 去重
        unique_results = []
        url_set = set()
        for result in sorted_results:
            if result["url"] not in url_set:
                url_set.add(result["url"])
                unique_results.append(result)

        # 限制返回数量
        final_results = unique_results[:num_results]

        return {
            "status": "COMPLETED",
            "message": f"成功提取{len(keyword_info['search_queries'])}个化工查询，获取到{len(final_results)}条结果",
            "keyword_info": keyword_info,
            "search_results": final_results
        }

    def _calculate_chemical_relevance(self, result: Dict, keywords: List[str]) -> int:
        """计算化工相关性分数"""
        title = result.get("title", "").lower()
        brief = result.get("brief", "").lower()
        content = title + " " + brief

        score = 0

        # 化工专有词加分
        for keyword in keywords:
            if keyword in content:
                score += 3
            elif any(word in content for word in self.keyword_extractor.chemical_terms if keyword in word):
                score += 2

        # 标题包含关键词比简介包含更重要
        for keyword in keywords:
            if keyword in title:
                score += 2

        # 化工相关网站加分
        chemical_domains = ['chem', '化工', '化学', '材料', '石油', '石化', '煤化工']
        url = result.get("url", "")
        if any(domain in url for domain in chemical_domains):
            score += 2

        return score

    def search_bing_with_chemical_filter(self, keyword: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        化工领域专用搜索，带过滤

        Args:
            keyword: 搜索关键词
            num_results: 返回结果数量

        Returns:
            化工相关搜索结果
        """
        search_results = []
        processed_urls = set()

        try:
            # 为化工搜索添加特定参数
            bing_url = (
                f"https://cn.bing.com/search?"
                f"q={requests.utils.quote(keyword)}&count={num_results * 3}&mkt=zh-CN&qs=HS&sc=10-0&cvid="
                f"&FORM=QBLH&sp=1"  # 添加更多参数
            )

            # 化工搜索可以稍微快一点
            time.sleep(random.uniform(1, 2))

            response = requests.get(
                bing_url,
                headers=self.default_headers,
                timeout=20,
                verify=False,
                allow_redirects=True
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # 化工领域可能更关注百科、论文、专利等
            result_selectors = [
                'div#b_results div.b_algo',
                'li.b_algo',
                'div#b_results div.mb-12',
                'div.b_entity',
                'div.b_answer',
            ]

            result_items = []
            for selector in result_selectors:
                result_items.extend(soup.select(selector))

            # 解析结果
            for result_item in result_items:
                if len(search_results) >= num_results:
                    break

                title_tag = result_item.select_one('h2 a') or result_item.select_one('a[href] h2')
                if not title_tag:
                    continue

                real_url = title_tag.get('data-href', '') or title_tag.get('href', '')
                if not real_url or not real_url.startswith('http') or real_url in processed_urls:
                    continue

                title_text = title_tag.get_text(strip=True)
                brief_tag = result_item.select_one('div.b_caption p') or result_item.select_one(
                    'div.b_lineclamp2') or result_item.select_one('div.b_lineclamp3')
                brief_text = brief_tag.get_text(strip=True) if brief_tag else ""

                if not title_text:
                    continue

                # 化工领域过滤：优先选择专业网站
                url_lower = real_url.lower()
                is_chemical_site = any(site in url_lower for site in [
                    'baike.baidu', 'wiki', 'chem', '化工', '化学', '材料', '知网', '万方', '专利'
                ])

                # 如果不是化工相关网站，但我们已经有很多结果，可以跳过
                if not is_chemical_site and len(search_results) >= max(2, num_results // 2):
                    continue

                search_results.append({
                    "title": title_text,
                    "url": real_url,
                    "brief": brief_text if brief_text else "无简介",
                    "search_query": keyword,
                    "is_chemical_site": is_chemical_site
                })
                processed_urls.add(real_url)

            logging.info(f"化工搜索完成：关键词「{keyword}」，获取到{len(search_results)}条结果")

        except Exception as e:
            logging.error(f"化工搜索失败：{str(e)}")

        return search_results

    def process_chemical_task(self,
                              task_description: str,
                              task_name: Optional[str] = None,
                              num_results: int = 3,
                              **kwargs) -> Dict[str, Any]:
        """
        处理化工领域任务

        Args:
            task_description: 化工任务描述
            task_name: 任务名称
            num_results: 需要抓取的结果数量
            **kwargs: 其他参数

        Returns:
            处理结果
        """
        task_id = f"chem_task_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        if not task_name:
            # 从化工任务描述生成任务名称
            keywords = self.keyword_extractor.extract_chemical_keywords(task_description, num_keywords=2)
            task_name = f"{'_'.join(keywords)}_化工搜索" if keywords else "化工信息搜索"

        # 1. 化工内容搜索
        search_result = self.search_chemical_content(task_description, num_results)

        if search_result["status"] != "COMPLETED" or not search_result["search_results"]:
            return {
                "status": "FAILED",
                "message": search_result.get("message", "化工搜索失败"),
                "task_id": task_id,
                "task_name": task_name,
                "search_result": search_result
            }

        # 2. 构造化工抓取任务
        tasks = []
        for idx, result in enumerate(search_result["search_results"][:num_results]):
            tasks.append({
                "task_id": f"{task_id}_{idx + 1}",
                "task_name": f"{task_name}_结果{idx + 1}",
                "task_description": f"抓取化工信息：{result['title']}",
                "generate_figure": False,
                "generate_table": kwargs.get("generate_table", False),
                "use_resources": [result['url']],
                "source_search_query": result.get("search_query", ""),
                "source_title": result['title'],
                "is_chemical_site": result.get("is_chemical_site", False)
            })

        # 3. 批量抓取
        batch_result = self.batch_process(tasks, **kwargs)

        # 4. 汇总结果
        final_result = {
            "status": "COMPLETED",
            "task_id": task_id,
            "task_name": task_name,
            "original_task_description": task_description,
            "task_type": search_result["keyword_info"]["task_type"],
            "search_summary": {
                "extracted_keywords": search_result["keyword_info"]["extracted_keywords"],
                "search_queries": search_result["keyword_info"]["search_queries"],
                "total_results_found": len(search_result["search_results"]),
                "chemical_sites_count": sum(
                    1 for r in search_result["search_results"] if r.get("is_chemical_site", False))
            },
            "scrape_results": batch_result,
            "total_processed": len(tasks)
        }

        return final_result

    # ---------------------- 原有方法（保持兼容性，但优化化工处理） ----------------------

    def _parse_dynamic_page(self, url: str, headers: dict) -> tuple:
        """动态页面解析"""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
                )
                context = browser.new_context(
                    user_agent=headers.get("User-Agent"),
                    viewport={"width": 1920, "height": 1080},
                    locale="zh-CN",
                    timezone_id="Asia/Shanghai"
                )
                context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
                """)
                page = context.new_page()
                page.set_extra_http_headers({k: v for k, v in headers.items() if k != "User-Agent"})

                page.goto(url, wait_until="networkidle", timeout=60000)
                page.wait_for_load_state("domcontentloaded")

                for _ in range(3):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(random.randint(1000, 2000))
                page.wait_for_timeout(3000)

                title = page.title()
                html_content = page.content()
                browser.close()

                soup = BeautifulSoup(html_content, "html.parser")
                for a_tag in soup.find_all('a'):
                    a_tag.decompose()
                for img_tag in soup.find_all('img'):
                    img_tag.decompose()

                content_parts = self._extract_main_content(soup)
                content = '\n\n'.join(content_parts)
                return self._filter_non_text_content(title), self._filter_non_text_content(content)

        except Exception as e:
            logging.error(f"动态页面解析失败：{str(e)}")
            return "", f"动态页面解析失败：{str(e)}"

    def _parse_static_page(self, url: str, headers: dict) -> tuple:
        """静态页面解析"""
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=15,
                verify=False,
                allow_redirects=True
            )
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all('a'):
                a_tag.decompose()
            for img_tag in soup.find_all('img'):
                img_tag.decompose()

            title_tag = soup.find("title") or soup.find("h1") or soup.find("h2")
            title = title_tag.get_text().strip() if title_tag else ""

            content_parts = self._extract_main_content(soup)
            if not content_parts:
                body_tag = soup.find('body')
                if body_tag:
                    content_parts = [
                        line.strip() for line in body_tag.get_text().splitlines()
                        if line.strip() and len(line.strip()) > 15
                    ]

            content = '\n\n'.join(content_parts)
            return self._filter_non_text_content(title), self._filter_non_text_content(content)

        except Exception as e:
            logging.error(f"静态页面解析失败：{str(e)}")
            return "", f"静态页面解析失败：{str(e)}"

    def _extract_main_content(self, soup: BeautifulSoup) -> List[str]:
        """提取页面主要内容"""
        main_selectors = [
            'article', 'main', 'div[class*="content"]', 'div[class*="article"]',
            'div[class*="main"]', 'section[class*="content"]', 'div[id*="content"]'
        ]
        content_parts = []

        for selector in main_selectors:
            for element in soup.select(selector):
                for p in element.find_all('p', recursive=True):
                    text = p.get_text(strip=True)
                    if text and len(text) > 10:
                        content_parts.append(text)
        return content_parts

    def _execute_scraping(self, url: str, headers: dict, **kwargs) -> tuple:
        """执行抓取逻辑"""
        firecrawl_success = False
        final_title = ""
        cleaned_content = ""

        if not kwargs.get("use_local_parse"):
            try:
                response = requests.post(
                    url=kwargs["api_endpoint"],
                    json={"url": url, "formats": ["markdown"]},
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {kwargs['api_key']}"
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()

                if result.get("success"):
                    data = result.get("data", {})
                    final_title = self._filter_non_text_content(data.get("title", ""))
                    cleaned_content = self._filter_non_text_content(data.get("markdown", ""))
                    firecrawl_success = True
            except Exception as e:
                logging.warning(f"Firecrawl接口调用失败，切换到本地解析：{str(e)}")

        if not firecrawl_success:
            if kwargs.get("use_dynamic_parse"):
                final_title, cleaned_content = self._parse_dynamic_page(url, headers)
                if "解析失败" in cleaned_content or not cleaned_content:
                    final_title, cleaned_content = self._parse_static_page(url, headers)
            else:
                final_title, cleaned_content = self._parse_static_page(url, headers)

        return final_title, cleaned_content

    def process_task(self,
                     task: Dict[str, Any],
                     messages: List[Dict[str, Any]],
                     cursor: int,
                     output_dir: str = None,
                     use_local_parse: bool = True,
                     use_dynamic_parse: bool = True,
                     api_endpoint: str = "http://localhost:3002/v1/scrape",
                     api_key: str = "local") -> Dict[str, Any]:
        """
        处理单个抓取任务（化工领域优化版）
        """
        output_dir = output_dir or self.output_dir
        current_result: Dict[str, Any] = {
            "task_id": task.get("task_id", f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            "section_name": task.get("task_name", "化工网页抓取内容"),
            "text_output": "",
            "status": "FAILED",
            "tables": [],
            "figures": [],
            "citations": [],
            "sources_used": task.get("use_resources", [])
        }

        # 检查是否有直接提供的URL
        use_resources = task.get("use_resources", [])

        # 化工领域：如果没有直接提供URL，但有任务描述，则使用化工专用搜索
        if not use_resources:
            task_description = task.get("task_description", "")
            if task_description:
                logging.info(f"化工任务{task.get('task_id')}没有指定URL，使用化工专用搜索：{task_description[:100]}...")

                # 使用化工专用搜索
                search_result = self.search_chemical_content(task_description, num_results=1)

                if search_result["status"] == "COMPLETED" and search_result["search_results"]:
                    use_resources = [search_result["search_results"][0]["url"]]
                    logging.info(f"化工搜索到URL：{use_resources[0]}")

                    # 更新任务信息
                    current_result["auto_searched"] = True
                    current_result["search_query_used"] = search_result["keyword_info"]["search_queries"][0] if \
                    search_result["keyword_info"]["search_queries"] else ""
                    current_result["original_task_description"] = task_description
                    current_result["task_type"] = search_result["keyword_info"]["task_type"]
                else:
                    error_msg = "无法从化工任务描述中搜索到有效的URL"
                    messages.append({
                        "role": "assistant",
                        "content": {
                            "from": "Worker",
                            "to": "Verifier",
                            "type": "WORKER_ERROR",
                            "error": error_msg
                        }
                    })
                    current_result["text_output"] = error_msg
                    logging.error(f"{current_result['task_id']} - {error_msg}")
                    return {"messages": messages, "current_result": current_result}
            else:
                error_msg = "任务参数缺失：use_resources为空且没有任务描述"
                messages.append({
                    "role": "assistant",
                    "content": {
                        "from": "Worker",
                        "to": "Verifier",
                        "type": "WORKER_ERROR",
                        "error": error_msg
                    }
                })
                current_result["text_output"] = error_msg
                logging.error(f"{current_result['task_id']} - {error_msg}")
                return {"messages": messages, "current_result": current_result}

        target_url = use_resources[0]
        messages.append({
            "role": "assistant",
            "content": {
                "from": "Worker",
                "to": "Verifier",
                "type": "WORKER_PROGRESS",
                "current_section": current_result["section_name"],
                "status": "开始化工内容爬取",
                "url": target_url,
                "cursor": cursor
            }
        })

        try:
            # 执行抓取
            final_title, cleaned_content = self._execute_scraping(
                url=target_url,
                headers=self.default_headers,
                use_local_parse=use_local_parse,
                use_dynamic_parse=use_dynamic_parse,
                api_endpoint=api_endpoint,
                api_key=api_key
            )

            # 化工领域内容整合
            task_type = current_result.get("task_type", "化工信息")
            chemical_keywords = self.keyword_extractor.extract_chemical_keywords(cleaned_content, num_keywords=5)

            final_text = f"""# {final_title}

## 化工内容元数据
- 任务ID：{current_result['task_id']}
- 任务类型：{task_type}
- 来源URL：{target_url}
- 爬取时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 任务描述：{task.get('task_description', '无')}
- 提取关键词：{', '.join(chemical_keywords) if chemical_keywords else '无'}

## 核心化工内容
{cleaned_content}
"""

            # 保存MD文件
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            os.makedirs(output_dir, exist_ok=True)
            safe_title = "".join(c for c in final_title if c.isalnum() or c in "._-")[:20]
            md_file_path = os.path.join(output_dir, f"chem_{timestamp}_{safe_title}.md")
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(final_text)

            # 更新结果状态
            is_success = "解析失败" not in cleaned_content and bool(cleaned_content)
            current_result.update({
                "text_output": final_text,
                "status": "COMPLETED" if is_success else "FAILED",
                "sources_used": use_resources + [md_file_path],
                "chemical_keywords": chemical_keywords
            })

            # 生成完成消息
            messages.append({
                "role": "assistant",
                "content": {
                    "from": "Worker",
                    "to": "Verifier",
                    "type": "WORKER_RESULT",
                    "current_section": current_result["section_name"],
                    "completed_sections": [current_result["section_name"]],
                    "progress": 1.0,
                    "result_status": current_result["status"]
                }
            })

            # 保存JSON结果
            json_file_path = os.path.join(output_dir, f"chem_{timestamp}_{safe_title}.json")
            self._save_json_result(
                {"messages": messages, "current_result": current_result},
                json_file_path
            )

            logging.info(f"{current_result['task_id']} - 化工内容处理完成，状态：{current_result['status']}")
            return {"messages": messages, "current_result": current_result}

        except Exception as e:
            error_msg = f"化工任务处理异常：{str(e)}"
            current_result["text_output"] = error_msg
            messages.append({
                "role": "assistant",
                "content": {
                    "from": "Worker",
                    "to": "Verifier",
                    "type": "WORKER_ERROR",
                    "error": error_msg
                }
            })
            logging.error(f"{current_result['task_id']} - {error_msg}")
            return {"messages": messages, "current_result": current_result}

    def batch_process(self,
                      tasks: List[Dict[str, Any]],
                      output_dir: str = None,
                      **kwargs) -> Dict[str, Any]:
        """批量处理化工任务"""
        output_dir = output_dir or self.output_dir
        from concurrent.futures import ThreadPoolExecutor, as_completed

        batch_id = f"chem_batch_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        results = []
        messages = []

        messages.append({
            "role": "assistant",
            "content": {
                "from": "Worker",
                "to": "Verifier",
                "type": "BATCH_PROGRESS",
                "status": "开始批量化工内容爬取",
                "total_tasks": len(tasks),
                "batch_id": batch_id
            }
        })

        max_workers = min(3, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.process_task, task, messages.copy(), idx, output_dir, **kwargs): (task, idx)
                for idx, task in enumerate(tasks)
            }

            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result["current_result"])
                except Exception as e:
                    logging.error(f"批量化工任务处理失败：{str(e)}")
                    results.append({
                        "task_id": f"error_chem_task_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "status": "FAILED",
                        "text_output": f"化工任务处理异常：{str(e)}"
                    })

        success_count = sum(1 for r in results if r["status"] == "COMPLETED")
        messages.append({
            "role": "assistant",
            "content": {
                "from": "Worker",
                "to": "Verifier",
                "type": "BATCH_COMPLETE",
                "batch_id": batch_id,
                "total_tasks": len(tasks),
                "success_tasks": success_count,
                "fail_tasks": len(tasks) - success_count,
                "completion_rate": f"{(success_count / len(tasks) * 100):.1f}%" if tasks else "0.0%"
            }
        })

        self._save_json_result(
            {"messages": messages, "results": results, "batch_id": batch_id},
            os.path.join(output_dir, f"chem_batch_summary_{batch_id}.json")
        )

        return {
            "batch_id": batch_id,
            "messages": messages,
            "results": results,
            "total_tasks": len(tasks),
            "success_tasks": success_count,
            "fail_tasks": len(tasks) - success_count
        }


# ---------------------- 测试 ----------------------
if __name__ == "__main__":
    scraper = WorkerScraper()

    print("=== 测试化工领域关键词提取 ===")

    # 化工领域测试案例
    chemical_test_cases = [
        "请帮我查找苯乙烯的制备工艺和主要用途",
        "我需要了解聚氯乙烯(PVC)的生产技术和市场现状",
        "查找关于催化剂在石油化工中的应用资料",
        "请提供甲醇制烯烃(MTO)技术的最新进展",
        "帮我搜索关于废水处理中膜分离技术的应用案例",
        "查找高密度聚乙烯(HDPE)的性能参数和加工方法"
    ]

    for desc in chemical_test_cases:
        print(f"\n化工任务描述: {desc}")

        # 提取关键词
        keywords = scraper.keyword_extractor.extract_chemical_keywords(desc, num_keywords=3)
        queries = scraper.keyword_extractor.generate_chemical_search_queries(desc, num_queries=2)

        print(f"提取关键词: {keywords}")
        print(f"生成查询: {queries}")

    # 测试完整的化工任务处理
    print("\n\n=== 测试完整化工任务处理 ===")

    test_task = "请帮我查找苯乙烯的制备工艺和主要用途"
    result = scraper.process_chemical_task(
        task_description=test_task,
        task_name="苯乙烯制备工艺查询",
        num_results=3,
        use_local_parse=False,
        use_dynamic_parse=True
    )

    if result["status"] == "COMPLETED":
        print(f"✅ 化工任务处理完成")
        print(f"任务ID: {result['task_id']}")
        print(f"任务类型: {result['task_type']}")
        print(f"提取关键词: {result['search_summary']['extracted_keywords']}")
        print(f"搜索查询: {result['search_summary']['search_queries']}")
        print(f"找到化工网站数量: {result['search_summary']['chemical_sites_count']}")
        print(f"抓取结果数: {result['total_processed']}")
    else:
        print(f"❌ 处理失败: {result.get('message', '未知错误')}")