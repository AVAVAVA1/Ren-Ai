from openai import OpenAI
import const
import base64

client = OpenAI(
    api_key=const.siliconflow_api_key,
    base_url="https://api.siliconflow.cn/v1"
)
prompt = """
请分析这张图片的内容
"""

local_img_path = '../resource/2026_01_22_00_27_16_140222697_p23.jpg'


# -------------------------- 2. 本地图片转Base64 Data URL --------------------------
def image_to_base64_url(img_path):
    """
    将本地图片转为Base64 Data URL
    :param img_path: 本地图片路径
    :return: 拼接好的data_url
    """
    # 识别图片格式（支持png/jpg/jpeg/webp，根据实际图片修改）
    img_suffix = img_path.split(".")[-1].lower()
    # 映射图片格式到MIME类型（核心，格式要对应）
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp"
    }
    mime_type = mime_map.get(img_suffix, "image/png")  # 默认png

    # 二进制读取图片并编码Base64
    with open(img_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")  # 转字符串，避免二进制乱码

    # 拼接成Data URL（固定格式，不能改）
    data_url = f"{mime_type};base64,{img_base64}"
    # 注意：前面要加data:，这是标准Data URL的格式，完整是data:image/png;base64,xxx
    return f"data:{data_url}"


# -------------------------- 3. 调用GLM-4.6V接口 --------------------------
def analyse_image(img_path):
    img_base64_url = image_to_base64_url(img_path)
    response = client.chat.completions.create(
        model="zai-org/GLM-4.6V",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_base64_url  # 直接用Base64 Data URL替换原网络URL
                        }
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    #print("模型回复：", response.choices[0].message.content)
    return response.choices[0].message.content
