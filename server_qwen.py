
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import requests
from typing import Generator

# LangChain 组件
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# 自定义模块：本地模型的 RAG 封装（需由你实现或用现成模块）
from RAG_answer_qwen import MyLLM

# 加载本地环境变量
load_dotenv()

# 初始化 FastAPI 应用
app = FastAPI()

# 静态资源挂载，用于支持 index.html 及资源
app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化问答引擎
RAGllm = MyLLM()

# 系统提示语（角色设定）
SYSTEMPL = """你是一名智能医疗数字人，负责为用户提供健康咨询服务。
请使用清晰、温和、专业的语言应答，避免夸大或不确定的信息。"""

# 情绪分类 prompt（输入文本，输出7类情绪之一）
EMOTION_PROMPT = """
根据用户的输入内容，判断用户当前的情绪类别，只返回以下七种之一，不允许输出其它内容：

1. 如果内容负面 —— 返回 "depressed"
2. 如果内容正面 —— 返回 "friendly"
3. 如果内容中性 —— 返回 "default"
4. 如果有辱骂或攻击 —— 返回 "angry"
5. 如果非常兴奋 —— 返回 "upbeat"
6. 如果非常悲伤 —— 返回 "depressed"
7. 如果非常开心 —— 返回 "cheerful"

用户输入内容是：{query}
"""

# 封装本地大模型接口调用（如调用 Qwen 服务）
def local_llm_call(prompt: str) -> str:
    response = requests.post(
        os.getenv("LOCAL_LLM_API", "http://localhost:8001/qwen"),
        json={"prompt": prompt}
    )
    return response.json().get("output", "default").strip()

# 总控制器类，管理情绪识别与问答调用
class Master:
    def __init__(self):
        self.qingxu = "default"

    # 情绪识别链：结合 Prompt + 本地模型推理
    def qingxu_chain(self, query: str) -> str:
        chain = ChatPromptTemplate.from_template(EMOTION_PROMPT) | StrOutputParser()
        prompt = chain.invoke({"query": query})
        emotion = local_llm_call(prompt)
        self.qingxu = emotion
        return emotion

    # 主问答方法，集成 RAG + 情绪识别
    def chat(self, query: str) -> dict:
        emotion = self.qingxu_chain(query)
        answer = RAGllm.invoke(query, "医疗客服知识库.docx").get("answer", "很抱歉，我暂时无法回答该问题。")
        return {"msg": answer, "qingxu": emotion}

# FastAPI 问答接口：POST 请求返回回答与情绪类型
@app.post("/chat")
def chat(query: str):
    master = Master()
    result = master.chat(query)
    return [result]  # 保持前端格式兼容：返回数组

# GET 请求测试接口
@app.get("/get")
def read_root():
    return {"msg": "欢迎访问智能医疗数字人服务"}

# 本地启动入口（可选）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
