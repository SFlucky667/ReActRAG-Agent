from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# 加载 .env 文件里的环境变量
load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    max_tokens=10
)

# 测试调用
resp = llm.invoke("你好")
print(resp.content)