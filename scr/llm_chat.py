from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


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
        result = self.llm.invoke(input_text)
        self.new_message = result
        return result

    def continuous_chat(self, input_text):
        # 添加用户消息
        self.messages.append({"role": "user", "content": input_text})

        response = self.llm.invoke(self.messages)

        self.messages.append({"role": "assistant", "content": response.content})
        self.new_message = response.content
        return response.content

    def structured_chat(self, input_text):
        """专门用于结构化输出的聊天"""
        result = self.chain.invoke({"text": input_text})
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

        final_result = chain.invoke({"text": result})
        print(final_result.res)
        return final_result
