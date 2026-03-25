import httpx
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from openai import APIConnectionError
from pydantic import BaseModel, Field
from typing import Literal, Optional


def message_to_text(msg) -> str:
    """从 invoke 返回的 AIMessage / BaseMessage 中取出可读正文（勿用 str(msg)，会得到对象字段串）。"""
    if msg is None:
        return ""
    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text")
                parts.append(str(t) if t is not None else str(block))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(msg)


def _is_upstream_connection_error(exc: BaseException) -> bool:
    cur: Optional[BaseException] = exc
    depth = 0
    while cur is not None and depth < 12:
        if isinstance(cur, APIConnectionError):
            return True
        if isinstance(cur, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
            return True
        if "connection error" in str(cur).lower():
            return True
        cur = cur.__cause__
        depth += 1
    return False


class LlmChat:
    def __init__(self, model_name, temperature, model_provider, base_url, api_key,
                 pydantic_object=None, prompt_template=None):
        self.model_name = model_name
        self.model_provider = model_provider
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.llm = init_chat_model(
            model=self.model_name,
            temperature=self.temperature,
            model_provider=self.model_provider,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.messages = []
        self.new_message = None

        # 如果提供了 Pydantic 对象，就创建解析器
        self.parser = PydanticOutputParser(pydantic_object=pydantic_object) if pydantic_object else None

        # 设置默认提示模板
        if prompt_template is None:
            if pydantic_object:
                format_instructions = self.parser.get_format_instructions() if self.parser else ""
                self.prompt_template = PromptTemplate(
                    template="""请处理以下文本：
                    输入文本：{text}
                    格式要求：{format_instructions}
                    输出：""",
                    input_variables=["text"],
                    partial_variables={"format_instructions": format_instructions}
                )
            else:
                # 普通对话模板
                self.prompt_template = PromptTemplate(
                    template="{text}",
                    input_variables=["text"]
                )
        else:
            self.prompt_template = prompt_template

        # 创建链
        if self.parser:
            self.chain = self.prompt_template | self.llm | self.parser
        else:
            self.chain = self.prompt_template | self.llm

    def singe_chat(self, input_text):
        try:
            result = self.llm.invoke(input_text)
        except Exception as e:
            if _is_upstream_connection_error(e):
                raise RuntimeError(
                    "无法连接到大模型 API（Connection error）。请检查：① 本机能否访问当前 base_url；"
                    "② 是否需代理（系统代理对 Python 未必生效，可尝试 VPN 或 HTTP 代理环境变量）；"
                    "③ 在 server/.env 中设置 LLM_API_BASE 为可用的兼容 OpenAI 协议的地址。"
                    f" 当前 base_url：{self.base_url}"
                ) from e
            raise
        self.new_message = result
        return result

    def continuous_chat(self, input_text):
        # 添加用户消息
        self.messages.append({"role": "user", "content": input_text})

        try:
            response = self.llm.invoke(self.messages)
        except Exception as e:
            if _is_upstream_connection_error(e):
                raise RuntimeError(
                    "无法连接到大模型 API（Connection error）。请检查网络、代理，或在 server/.env 设置 LLM_API_BASE。"
                    f" 当前 base_url：{self.base_url}"
                ) from e
            raise

        self.messages.append({"role": "assistant", "content": response.content})
        self.new_message = response.content
        return response.content

    def structured_chat(self, input_text):
        """专门用于结构化输出的聊天"""
        try:
            result = self.chain.invoke({"text": input_text})
        except Exception as e:
            if _is_upstream_connection_error(e):
                raise RuntimeError(
                    "无法连接到大模型 API（Connection error）。请检查网络、代理，或在 server/.env 设置 LLM_API_BASE。"
                    f" 当前 base_url：{self.base_url}"
                ) from e
            raise
        self.new_message = result
        return result

    def clear_messages(self):
        self.messages = []

    def change_prompt_template(self, prompt_template):
        self.prompt_template = prompt_template
        if self.parser:
            self.chain = self.prompt_template | self.llm | self.parser
        else:
            self.chain = self.prompt_template | self.llm

    def change_parser(self, pydantic_object):
        self.parser = PydanticOutputParser(pydantic_object=pydantic_object)
        self.chain = self.prompt_template | self.llm | self.parser

    def change_temperature(self, temperature):
        self.temperature = temperature
        self.llm = init_chat_model(
            model=self.model_name,
            temperature=self.temperature,
            model_provider=self.model_provider,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        if self.parser:
            self.chain = self.prompt_template | self.llm | self.parser
        else:
            self.chain = self.prompt_template | self.llm

    def evaluate_result(self, result, result_type):
        class EvaluationResult(BaseModel):
            res: Literal["Perfect", "Good", "OK", "Just so so", "Bad"] = Field(description="最终的评判结果"
                                                                                           "Perfect 是非常好，无需改动了"
                                                                                           "Good 是很好第二档，可以进行一定修改"
                                                                                           "OK 是还行第三档，有缺点"
                                                                                           "Just so so 是一般第四档"
                                                                                           "Bad 是很差第五档")
            reason_and_advise: str = Field(description="得出评价的理由及可能的改进方向")

        parser = PydanticOutputParser(pydantic_object=EvaluationResult)
        prompt = PromptTemplate(
            template=f'完成{result_type}的评价.' + """你是一个专业的作品评价家。请根据所给的内容(大纲，文章，剧本等)完成评价.
                        评价的要求务必严格，给出的改进建议务必详细,确保下次评估是达到更好的水准.

            要鉴赏的文本：{text}

            请严格按照以下格式输出,不要添加任何额外说明：：
            {format_instructions}

            输出：""",
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.llm | parser

        try:
            final_result = chain.invoke({"text": message_to_text(result)})
        except Exception as e:
            if _is_upstream_connection_error(e):
                raise RuntimeError(
                    "无法连接到大模型 API（Connection error）。请检查网络、代理，或在 server/.env 设置 LLM_API_BASE。"
                    f" 当前 base_url：{self.base_url}"
                ) from e
            raise
        print(final_result.res)
        return final_result