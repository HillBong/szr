
import requests
import os
from langchain_core.language_models import BaseLanguageModel

class LocalQwenLLM(BaseLanguageModel):
    """
    用于替代 OpenAI 的本地 Qwen 模型封装，兼容 LangChain 接口。
    """

    def __init__(self):
        # 本地部署的微调模型服务接口地址，可通过环境变量配置
        self.api_url = os.getenv("LOCAL_LLM_API", "http://localhost:8001/qwen")

    def _call(self, prompt: str, **kwargs) -> str:
        try:
            response = requests.post(
                self.api_url,
                json={"prompt": prompt},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("output", "(模型无响应)")
        except Exception as e:
            return f"(本地模型调用出错: {str(e)})"

    def invoke(self, input: str, **kwargs) -> str:
        """
        LangChain 标准接口，接收 prompt 输入并返回模型响应。
        """
        return self._call(input)
