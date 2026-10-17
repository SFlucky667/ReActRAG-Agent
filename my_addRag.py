from dotenv import load_dotenv
import os
import requests
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 尝试导入chromadb（一个轻量级本地向量数据库）
# 如果没安装，代码会自动降级为"关键词匹配"模式
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

load_dotenv();
# 最大思考，行动轮数
MAX_STEPS=5;

# ============================================================
# 第一步：准备知识库数据
# 实际项目中，这些数据可以来自：数据库、API、爬虫、文档解析
# 这里用模拟的资讯数据来演示
# ============================================================

# 模拟的资讯/知识库文档（你可以替换成真实数据）
KNOWLEDGE_DOCS = [
    "2024年1月15日，哈尔滨市迎来强降雪天气，全市降水量达到大到暴雪级别，"
    "气温降至零下25摄氏度，相关部门启动冰雪应急预案，提醒市民注意防寒保暖。"
    "哈尔滨市交通局启动除雪预案，主城区主要道路实行机械+人工联合清雪。",

    "2024年1月14日，北京市公布2023年经济数据，全年GDP同比增长5.2%，"
    "突破4.5万亿元。其中数字经济增加值占比超过40%，"
    "成为经济增长的核心驱动力。高技术制造业增加值增长10.3%。",

    "2024年1月13日，上海市发布《人工智能产业发展行动计划（2024-2026）》，"
    "计划三年内培育100家百亿级人工智能企业，"
    "建设张江人工智能岛，打造世界级AI产业集群。"
    "政策涵盖算力补贴、人才落户、场景开放等方面。",

    "2024年1月12日，广州市天河区某科技公司办公楼发生火灾，"
    "消防部门接警后迅速到场处置，约30分钟后火势被扑灭。"
    "暂无人员伤亡报告，起火原因正在调查中。"
    "该楼栋部分区域烟雾较大，周边道路临时交通管制。",

    "2024年1月10日，深圳市宣布新增新能源汽车充电桩5000个，"
    "全市累计充电桩数量超过50万个，居全国城市首位。"
    "深圳市新能源汽车渗透率已超过60%，"
    "计划2025年实现公交车100%电动化。",

    "2024年1月8日，中国人民银行宣布下调金融机构存款准备金率0.5个百分点，"
    "释放长期资金约1万亿元。这是2023年以来的第二次降准，"
    "旨在支持实体经济发展，降低社会综合融资成本。"
    "市场预计贷款市场报价利率(LPR)也可能随之下调。",

    "2024年1月5日，国家统计局发布数据，12月份全国CPI同比上涨0.5%，"
    "PPI同比下降2.3%。12月份，食品价格下降1.2%，"
    "其中猪肉价格下降8.5%；能源价格下降3.1%。"
    "整体来看，经济保持温和复苏态势，通胀压力可控。",

    "2024年1月3日，国际奥委会宣布黑龙江省成功申办2025年"
    "第九届亚洲冬季运动会。赛事将于2025年2月7日至14日"
    "在哈尔滨和亚布力举行，共设6个大项64个小项。"
    "这是中国第三次举办亚冬会，首次在主城哈尔滨举办。",

    "2024年1月2日，全国铁路春运正式启动，预计发送旅客4.8亿人次，"
    "同比增长约10%。铁路部门加开夜间高铁列车200余列，"
    "新增临时旅客列车150对。"
    "民航预计春运期间运送旅客7900万人次，同比增长12%。",

    "2024年1月1日，《区域全面经济伙伴关系协定》(RCEP)全面生效三周年，"
    "中国与RCEP成员国贸易额占中国外贸总额比重提升至30.8%。"
    "其中对东盟进出口增长8.5%，对日本增长6.2%。"
    "RCEP区域内中间品贸易活跃度显著提升。",
]

# ============================================================
# 第二步：初始化知识库（向量数据库）
# ============================================================

# 全局知识库实例
kb_collection = None

def init_knowledge_base():
 global kb_collection

 if HAS_CHROMA:
   client = chromadb.Client(Settings())
 try:
            collection = client.get_collection("news_knowledge")
            print("[知识库] 使用已有知识库集合")
 except Exception:
            collection = client.create_collection("news_knowledge")
            # 将资讯文档加入知识库
            # chromadb会自动调用内置的sentence-transformers模型
            # 把每段文字转成向量并存储
            collection.add(
                documents=KNOWLEDGE_DOCS,
                ids=[str(i) for i in range(len(KNOWLEDGE_DOCS))]
            )
            print(f"[知识库] 已初始化，共存入 {len(KNOWLEDGE_DOCS)} 篇资讯")
        
 kb_collection = collection
 return "chroma"  # 返回使用的模式
 else:
    # === 模式B：降级为关键词匹配（无需额外安装） ===
    kb_collection = {
            "docs": KNOWLEDGE_DOCS,
            "mode": "keyword"
        }
    print("[知识库] chromadb未安装，已降级为关键词匹配模式")
    print("[提示] 运行 pip install chromadb 可启用语义搜索")
    return "keyword"
# 启动时初始化知识库
rag_mode = init_knowledge_base()

# 第三步：RAG检索函数
def rag_search(query: str, top_k: int = 2) -> str:
    """
    RAG检索函数：从知识库中检索与查询最相关的资讯。
    
    参数:
        query: 用户的查询问题
        top_k: 返回最相关的k条结果
    
    返回:
        检索到的资讯文本，格式化为大模型可读的格式
    """
    if rag_mode == "chroma":
        # 向量搜索：chromadb会把query也转成向量，
        # 然后计算向量之间的余弦相似度，返回最相似的文档
        results = kb_collection.query(
            query_texts=[query],
            n_results=top_k
        )
    if results['documents'] and results['documents'][0]:
            # 格式化返回结果
            docs = results['documents'][0]
            # 加上相关性分数，方便调试
            scores = results.get('distances', [[]])[0]
            formatted = []
            for i, (doc, score) in enumerate(zip(docs, scores)):
                formatted.append(
                    f"[资讯{i+1}] (相似度{1-score:.2f}) {doc}"
                )
            return "【检索到的相关资讯】\n" + "\n".join(formatted)
        
    return "知识库中未找到与您的问题相关的资讯"
    else:
        # 关键词匹配（降级模式）
        # 简单实现：统计query中的关键词在各文档中出现的频率
    import re
        # 提取query中的关键词（中文按字符，英文按单词）
    query_terms = set(re.findall(r'[\u4e00-\u9fa5]{2,}|\w{3,}', query.lower()))
        
    if not query_terms:
            return "知识库中未找到与您的问题相关的资讯"
        
        # 计算每篇文档的关键词匹配得分
    scored_docs = []
    for i, doc in enumerate(KNOWLEDGE_DOCS):
            doc_lower = doc.lower()
            match_count = sum(1 for term in query_terms if term in doc_lower)
            if match_count > 0:
                scored_docs.append((match_count, i, doc))
        
        # 按匹配度降序排列，取前top_k条
    scored_docs.sort(key=lambda x: x[0], reverse=True)
        
    if scored_docs:
            formatted = []
            for rank, (count, idx, doc) in enumerate(scored_docs[:top_k]):
                formatted.append(
                    f"[资讯{rank+1}] (匹配{count}个关键词) {doc}"
                )
            return "【检索到的相关资讯】\n" + "\n".join(formatted)
        
    return "知识库中未找到与您的问题相关的资讯"


# 【RAG工具】供Agent调用的资讯查询工具

def rag_tool(query: str) -> str:
    """
    RAG知识库查询工具。
    Agent在需要回答资讯类问题时调用此工具。
    
    参数:
        query: 从用户问题中提取的查询关键词
    
    返回:
        检索到的相关资讯文本
    """
    return rag_search(query)

#1.定义Agent的状态
class AgentState(TypedDict):
    question:str
    thougnt:str
    action:str
    observation:str
    answer:str
    step:str

#2.天气查询工具
def get_weather(city:str)->str:
    city_coords={
        "哈尔滨": (45.75, 126.65),
        "北京": (39.90, 116.40),
        "上海": (31.23, 121.47),
        "广州": (23.13, 113.26),
        "深圳": (22.54, 114.05),
    }

    if city not in city_coords:
        return f"暂不支持查询【{city}】的天气"
    # 发请求
    lat,lon=city_coords[city]
    url = "https://api.open-meteo.com/v1/forecast"
    params={
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
    }

    try:
        resp=requests.get(url,params=params,timeout=5)
        resp.raise_for_status()
        data=resp.json() # 将resp的json文本解析成python字典
        current = data.get("current_weather", {})
        temp = current.get("temperature")
        windspeed = current.get("windspeed")
        weathercode = current.get("weathercode")
        return (
            f"{city}当前天气：温度{temp}℃，"
            f"风速{windspeed}km/h，天气代码{weathercode}"
        )
    except Exception as e:
        return f"查询失败：{str(e)}"

#3.计算器工具
def calculator(expression:str)->str:
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含不允许的字符"
    try:
        result=eval(expression)
        return f"{expression}={result}"
    except Exception as e:
        return f"计算失败：{str(e)}"


#4.初始化模型
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    max_tokens=10
)


#5.思考节点
def agent_think(state:AgentState):
    current_step=state.get("step",0)+1;
    #把变量填进一段模版提示词里，使其生成发给大模型的完整指令
    prompt=f"""
    你是智能体，根据用户问题，输出：
    Thougnt:你的思考
    action:可选get_weather,calculator,finish
    用户问题：{state['question']}
    之前观察：{state.get('observation','')}
    当前第{current_step}轮，最多允许{MAX_STEPS}轮。
    """

    #调用大模型,把prompt发给大模型，提取了纯文本的内容
    resp=llm.invoke(prompt).content

    #解析模型的回复
    thought=""
    action=""
    for line in resp.split("\n"):
        if line.startswith("Thought:"):
            thought=line.replace("Thougnt:","").strip()
        if line.startswith("Action:"):
            action=line.replace("Action:","").strip()
    return {
            "thought":thought,
            "action":action,
            "step":current_step
        }

# 6..工具执行节点
def agent_action(state:AgentState):
    action=state["state"]  #获取思考到的动作
    question=state["question"] #取出用户的原始问题

    if action=="get_weather":
        obs=get_weather("question")
    elif action=="calculator":
        obs=calculator("question")
    else:
        obs="无需调用工具"
    return {
        "observation":obs
    }

# 7..最终答案节点
def agent_answer(state:AgentState):
    prompt=f"""
    基于下面信息，回答用户问题。
    问题：{state['question']}
    思考：{state['thought']}
    观察:{state['observation']}
    """
    ans=llm.invoke(prompt).content
    return{
        "answer":ans
    }

# 8.路由函数(在LangGraph里，一个节点执行完后，下一步去哪，是由edge决定的，如果没有路由函数，图就不知道往哪走)
def route_action(state:AgentState):
    if state["action"]=="finish":
        return "answer"
    if state.get("step",0)>=MAX_STEPS:
        return "answer"

    return "action"

# 9.构建表（包含建图时的映射表-->手动注册）存在的意义：把前面的节点和路由函数组装成一个可运行的函数
graph=StateGraph(AgentState)
    #注册三个节点（左侧是节点名，右侧是节点函数）
graph.add_node("think",agent_think)
graph.add_node("action",agent_action)
graph.add_node("answer",agent_answer)

    #设置入口点
graph.set_empty_point("think")
graph.add_conditional_edges(
    "think",
    route_action,
    {"action":"action","answer":"answer"}
)
graph.add_edge("action","think")
graph.add_edge("answer",END)

agent=graph.compile()

#10.运行测试
if __name__=="__main__":
    print("=== 测试天气查询 ===")
    res=agent.invoke({
        "question":"查询哈尔滨当前天气，形式是今天的日期加今天的温度",
        "step":0
    })
    print("最终回答:",res["answer"])
    print("执行论述：",res.get("step"))

    print("=== 测试计算器 ===")
    res=agent.invoke({
        "question":"100*9",
        "step":0
    })
    print("最终回答:",res["answer"])
    print("执行论述：",res.get("step"))

