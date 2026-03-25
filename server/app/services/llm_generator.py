from typing import Type, TypeVar, Optional, Callable
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from app.services import const, llm_chat

T = TypeVar('T', bound=BaseModel)


class LLMGenerator:
    """通用的 LLM 生成器，封装评估和重试逻辑"""
    
    def __init__(
        self,
        pydantic_object: Type[T],
        result_type: str,
        model_name: Optional[str] = None,
        temperature: float = 1.0,
        model_provider: str = 'openai',
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt_template: Optional[PromptTemplate] = None,
        max_retries: int = 10
    ):
        self.pydantic_object = pydantic_object
        self.result_type = result_type
        self.model_name = model_name or const.llm_model
        self.temperature = temperature
        self.model_provider = model_provider
        self.base_url = base_url or const.llm_base_url
        if api_key is not None and str(api_key).strip():
            self.api_key = str(api_key).strip()
        else:
            self.api_key = str(const.api_key).strip() if const.api_key else ""
        self.max_retries = max_retries

        if not (self.api_key and str(self.api_key).strip()):
            raise ValueError(
                "未配置大模型 API Key：请在 server 目录下创建或编辑 .env，设置 DS_API_KEY=你的密钥 "
                "（也可使用 LLM_API_KEY）。启动方式见 start-backend.ps1。"
            )
        
        self.parser = PydanticOutputParser(pydantic_object=pydantic_object)
        
        if prompt_template is None:
            self.prompt_template = self._create_default_prompt()
        else:
            self.prompt_template = prompt_template
        
        self.llm = llm_chat.LlmChat(
            model_name=self.model_name,
            temperature=temperature,
            model_provider=model_provider,
            base_url=self.base_url,
            api_key=self.api_key,
            pydantic_object=pydantic_object,
            prompt_template=self.prompt_template,
        )
    
    def _create_default_prompt(self) -> PromptTemplate:
        """创建默认的提示词模板"""
        return PromptTemplate(
            template="""请根据以下要求生成内容：
            创作要求：{text}
            
            请严格按照以下格式输出,不要添加任何额外说明：
            {format_instructions}
            
            输出：""",
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def generate_with_retry(
        self,
        user_input: str,
        strict_model: bool = False,
        content_formatter: Optional[Callable[[str, str], str]] = None,
        final_temperature: float = 0.0
    ) -> T:
        """
        带评估和重试的生成方法
        
        Args:
            user_input: 用户输入或要求
            strict_model: 是否严格模式（只有 Perfect 才通过）
            content_formatter: 内容格式化函数，用于构建每次迭代的输入
            final_temperature: 最终结构化输出时的温度
        
        Returns:
            生成的结构化结果
        """
        retry_count = 0
        
        while retry_count < self.max_retries:
            if content_formatter:
                prev = llm_chat.message_to_text(self.llm.new_message) if self.llm.new_message else ""
                input_text = content_formatter(user_input, prev)
            else:
                input_text = user_input
            
            self.llm.singe_chat(input_text)
            
            eva = self.llm.evaluate_result(self.llm.new_message, self.result_type)
            
            if eva.res == "Perfect" or (eva.res == "Good" and not strict_model):
                print(f"生成成功，重试次数：{retry_count}")
                
                self.llm.change_temperature(final_temperature)
                tail = (
                    llm_chat.message_to_text(self.llm.new_message)
                    + "\n以上为最终确定的内容，按我要求的格式输出，不要进行额外删改，也不要有多余内容"
                )
                try:
                    return self.llm.structured_chat(tail)
                except OutputParserException:
                    # 模型常返回夹杂说明文字或非严格 JSON；再要一次纯 JSON
                    self.llm.singe_chat(
                        "你的上一份回复无法按 JSON schema 解析。"
                        "请只输出一个 JSON 对象（不要用 markdown 代码块包裹），"
                        "键名与 schema 完全一致；content 必须是字符串数组，每元素为一段剧本正文；"
                        "正文内换行须写成 \\n。"
                    )
                    return self.llm.structured_chat(
                        llm_chat.message_to_text(self.llm.new_message)
                        + "\n仅输出符合格式说明的 JSON，不要其它文字。"
                    )
            else:
                retry_count += 1
                self.llm.singe_chat(
                    f'完成内容的修改。原内容:{llm_chat.message_to_text(self.llm.new_message)}.\n用户原始要求（务必参考）{user_input}\n其他修改建议:{eva.reason_and_advise}'
                )
        
        raise Exception(f"生成失败，已达到最大重试次数 {self.max_retries}")
