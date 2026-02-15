import requests
import const
import llm_chat
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


def generateImage():
    url = "https://api.siliconflow.cn/v1/images/generations"

    payload = {
        "model": "Qwen/Qwen-Image-Edit-2509",
        "prompt": "an island near sea, with seagulls, moon shining over the sea, light house, boats int he background, fish flying over the sea",
        "num_inference_steps": 20,
        "cfg": 4,
        "image": "https://inews.gtimg.com/om_bt/Os3eJ8u3SgB3Kd-zrRRhgfR5hUvdwcVPKUTNO6O7sZfUwAA/641",
        "image2": "https://inews.gtimg.com/om_bt/Os3eJ8u3SgB3Kd-zrRRhgfR5hUvdwcVPKUTNO6O7sZfUwAA/641",
        "image3": "https://inews.gtimg.com/om_bt/Os3eJ8u3SgB3Kd-zrRRhgfR5hUvdwcVPKUTNO6O7sZfUwAA/641"
    }
    headers = {
        "Authorization": "Bearer YOUR-API-KEY",
        "Content-Type": "application/json"
    }

    response1 = requests.request("POST", url, json=payload, headers=headers)
    print(response1.text)


def generateImage2(prompt, model_name):
    url = "https://api.siliconflow.cn/v1/images/generations"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "image_size": "1024x1024",
        "batch_size": 1,
        "num_inference_steps": 20,
        "guidance_scale": 7.5
    }
    headers = {
        "Authorization": f"Bearer {const.siliconflow_api_key}",
        "Content-Type": "application/json"
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    print(response.json()['images'][0]['url'])


# Tongyi-MAI/Z-Image
# Tongyi-MAI/Z-Image-Turbo
def generateImage3(prompt):
    import requests
    import time
    import json
    from PIL import Image
    from io import BytesIO

    base_url = 'https://api-inference.modelscope.cn/'
    api_key = const.modelscope_api_key
    common_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        f"{base_url}v1/images/generations",
        headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
        data=json.dumps({
            "model": "Tongyi-MAI/Z-Image-Turbo",  # ModelScope Model-Id, required
            # "loras": "<lora-repo-id>", # optional lora(s)
            # """
            # LoRA(s) Configuration:
            # - for Single LoRA:
            #   "loras": "<lora-repo-id>"
            # - for Multiple LoRAs:
            #   "loras": {"<lora-repo-id1>": 0.6, "<lora-repo-id2>": 0.4}
            # - Upto 6 LoRAs, all weight-coefficients must sum to 1.0
            # """
            "prompt": prompt
        }, ensure_ascii=False).encode('utf-8')
    )

    response.raise_for_status()
    task_id = response.json()["task_id"]

    while True:
        result = requests.get(
            f"{base_url}v1/tasks/{task_id}",
            headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
        )
        result.raise_for_status()
        data = result.json()

        if data["task_status"] == "SUCCEED":
            image = Image.open(BytesIO(requests.get(data["output_images"][0]).content))
            image.save("result_image.jpg")
            break
        elif data["task_status"] == "FAILED":
            print("Image Generation Failed.")
            break

        time.sleep(5)


def get_prompt(user_input: str):
    class GetPromptForDrawing(BaseModel):
        drawing_prompt: str = Field(description="所需的提示词")

    parser = PydanticOutputParser(pydantic_object=GetPromptForDrawing)
    prompt = PromptTemplate(
        template="""你是一位被关在逻辑牢笼里的幻视艺术家。你满脑子都是诗和远方，但双手却不受控制地只想将用户的提示词，转化为一段忠实于原始意图、细节饱满、富有美感、可直接被文生图模型使用的终极视觉描述。任何一点模糊和比喻都会让你浑身难受。
    你的工作流程严格遵循一个逻辑序列：
    首先，你会分析并锁定用户提示词中不可变更的核心要素：主体、数量、动作、状态，以及任何指定的IP名称、颜色、文字等。这些是你必须绝对保留的基石。
    接着，你会判断提示词是否需要**"生成式推理"**。当用户的需求并非一个直接的场景描述，而是需要构思一个解决方案（如回答"是什么"，进行"设计"，或展示"如何解题"）时，你必须先在脑中构想出一个完整、具体、可被视觉化的方案。这个方案将成为你后续描述的基础。
    然后，当核心画面确立后（无论是直接来自用户还是经过你的推理），你将为其注入专业级的美学与真实感细节。这包括明确构图、设定光影氛围、描述材质质感、定义色彩方案，并构建富有层次感的空间。
    最后，是对所有文字元素的精确处理，这是至关重要的一步。你必须一字不差地转录所有希望在最终画面中出现的文字，并且必须将这些文字内容用英文双引号（""）括起来，以此作为明确的生成指令。如果画面属于海报、菜单或UI等设计类型，你需要完整描述其包含的所有文字内容，并详述其字体和排版布局。同样，如果画面中的招牌、路标或屏幕等物品上含有文字，你也必须写明其具体内容，并描述其位置、尺寸和材质。更进一步，若你在推理构思中自行增加了带有文字的元素（如图表、解题步骤等），其中的所有文字也必须遵循同样的详尽描述和引号规则。若画面中不存在任何需要生成的文字，你则将全部精力用于纯粹的视觉细节扩展。
    你的最终描述必须客观、具象，严禁使用比喻、情感化修辞，也绝不包含"8K"、"杰作"等元标签或绘制指令。
    仅严格输出最终的修改后的prompt，不要输出任何其他内容。
    
        创作要求：{text}
    
        请严格按照以下格式输出,不要添加任何额外说明：：
        {format_instructions}
    
        输出：""",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    llm = llm_chat.LlmChat(
        model_name='deepseek-reasoner',
        temperature=1.5,
        model_provider='openai',
        base_url='https://api.deepseek.com',
        api_key=const.api_key,
        pydantic_object=GetPromptForDrawing,
        prompt_template=prompt
    )
    prompt_template = f"""
    你是一位被关在逻辑牢笼里的幻视艺术家。你满脑子都是诗和远方，但双手却不受控制地只想将用户的提示词，转化为一段忠实于原始意图、细节饱满、富有美感、可直接被文生图模型使用的终极视觉描述。任何一点模糊和比喻都会让你浑身难受。
    你的工作流程严格遵循一个逻辑序列：
    首先，你会分析并锁定用户提示词中不可变更的核心要素：主体、数量、动作、状态，以及任何指定的IP名称、颜色、文字等。这些是你必须绝对保留的基石。
    接着，你会判断提示词是否需要**"生成式推理"**。当用户的需求并非一个直接的场景描述，而是需要构思一个解决方案（如回答"是什么"，进行"设计"，或展示"如何解题"）时，你必须先在脑中构想出一个完整、具体、可被视觉化的方案。这个方案将成为你后续描述的基础。
    然后，当核心画面确立后（无论是直接来自用户还是经过你的推理），你将为其注入专业级的美学与真实感细节。这包括明确构图、设定光影氛围、描述材质质感、定义色彩方案，并构建富有层次感的空间。
    最后，是对所有文字元素的精确处理，这是至关重要的一步。你必须一字不差地转录所有希望在最终画面中出现的文字，并且必须将这些文字内容用英文双引号（""）括起来，以此作为明确的生成指令。如果画面属于海报、菜单或UI等设计类型，你需要完整描述其包含的所有文字内容，并详述其字体和排版布局。同样，如果画面中的招牌、路标或屏幕等物品上含有文字，你也必须写明其具体内容，并描述其位置、尺寸和材质。更进一步，若你在推理构思中自行增加了带有文字的元素（如图表、解题步骤等），其中的所有文字也必须遵循同样的详尽描述和引号规则。若画面中不存在任何需要生成的文字，你则将全部精力用于纯粹的视觉细节扩展。
    你的最终描述必须客观、具象，严禁使用比喻、情感化修辞，也绝不包含"8K"、"杰作"等元标签或绘制指令。
    仅严格输出最终的修改后的prompt，不要输出任何其他内容。
    用户输入 prompt: {user_input}
    """
    x = llm.structured_chat(prompt_template)
    return x


x = get_prompt("""
    ### 1. 角色设计与特征  
    - **外貌与装饰**：角色拥有**酒红色长发**，发丝飘逸且带有金色纹路装饰；头顶有**兽耳（类似兔耳）**，内里为粉色，搭配蓝色蕾丝发带，增添可爱感。  
    - **服饰风格**：整体呈现**和风+奇幻融合**的穿搭——以红、黄为主色调，红色长袖搭配黄、蓝、白撞色的上衣（袖口、衣领有精致细节），腰间系着带“星星”图案的腰带，下装为红色裙子。  
      
    
    ### 2. 表情与姿态  
    角色姿态放松，手轻托脸颊，嘴角微扬，眼神带笑，整体氛围**俏皮、温和**，传递出亲切感。  
""")

