from dotenv import load_dotenv
import os
import requests
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict


load_dotenv();
# 最大思考，行动轮数
MAX_STEPS=5;

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

