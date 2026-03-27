import http.client
import json
import re
from pydantic import BaseModel, Field
from app.services import const, llm_chat, tools
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os
import time
import zipfile
import tempfile
import shutil
import requests
from typing import List, Optional
from urllib.parse import urlparse

# 与 get_running_pic 中用于文件名后缀的表情标识一致
EXPRESSION_LS = ['happy', 'depression', 'crying', 'smile', 'surprise', 'shy', 'tsundere','anger','sad']


def sanitize_character_name_for_path(name: str) -> str:
    """用于 public/sources/pic 下的子文件夹名。"""
    s = (name or "").strip()
    if not s:
        return "unnamed"
    for c in '<>:"/\\|?*\n\r\t':
        s = s.replace(c, '_')
    s = s.strip(' .')
    return s or "unnamed"


def sanitize_expression_for_filename(expr: str) -> str:
    """表情标签用作文件名的一部分。"""
    s = (expr or "").strip()
    if not s:
        return "expr"
    for c in '<>:"/\\|?*':
        s = s.replace(c, '_')
    return s


# 立绘保存名：{expression}_1.png（去背景逻辑同时认 {expression}.png 与 {expression}_{数字}.png）
_STAND_PIC_NUMERIC_SUFFIX = "_1"


def stand_pic_save_filename(expression: str) -> str:
    tag = sanitize_expression_for_filename(expression)
    return f"{tag}{_STAND_PIC_NUMERIC_SUFFIX}.png"


def list_stand_pic_png_paths(dir_path: str, expr: str) -> List[str]:
    """匹配 {expression}.png 或 {expression}_<数字>.png，返回完整路径（不区分大小写，已排序）。"""
    tag = re.escape(sanitize_expression_for_filename(expr))
    pat = re.compile(rf'^{tag}(_\d+)?\.png$', re.IGNORECASE)
    try:
        names = os.listdir(dir_path)
    except OSError:
        return []
    matched = [
        n
        for n in names
        if pat.match(n) and os.path.isfile(os.path.join(dir_path, n))
    ]
    matched.sort(key=str.lower)
    return [os.path.join(dir_path, n) for n in matched]


# 2037082428853981185
def run_RunningHub(workflowId: str, positive_prompt: str):
    api_key = const.running_hub_key
    conn = http.client.HTTPSConnection("www.runninghub.cn")
    payload = json.dumps({
        "apiKey": api_key,
        "workflowId": workflowId,
        "nodeInfoList": [
            {
                "nodeId": "48",
                "fieldName": "text",
                "fieldValue": positive_prompt
            },
            {
                "nodeId": "27",
                "fieldName": "value",
                "fieldValue": 4
            }
        ]
    })
    headers = {
        'Host': 'www.runninghub.cn',
        'Authorization': api_key,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/task/openapi/create", payload, headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    result_json = json.loads(data)
    print(result_json)
    if result_json["msg"] == "success":
        task_id = result_json['data']['taskId']
        return task_id
    print("task error:", result_json)
    return None


def download_image_from_url(
        image_url: str,
        save_dir: str = "./downloaded_images",
        save_filename: Optional[str] = None
) -> List[str]:
    """
    从URL下载资源（图片或ZIP压缩包）并保存到本地。
    - 如果是图片：直接保存，文件名可指定或自动提取。
    - 如果是ZIP压缩包：解压并提取其中所有图片文件，按 `base_name_{序号}.ext` 命名，保存到 save_dir。
      base_name 取自 save_filename（不含扩展名），若未提供则自动生成。
    :param image_url: 资源URL地址
    :param save_dir: 保存文件夹（不存在会自动创建）
    :param save_filename: 单张图片时直接作为文件名（可带扩展名）；
                          压缩包时作为基础名称（自动去除扩展名），用于拼接序号。
    :return: 所有保存图片的完整路径列表（失败时返回空列表）
    """
    if not image_url:
        print("❌ 错误：URL为空")
        return []

    os.makedirs(save_dir, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    temp_file = None
    temp_dir = None
    try:
        response = requests.get(image_url, headers=headers, stream=True, timeout=15)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_file = tmp.name
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)

        # 判断是否为ZIP
        is_zip = False
        content_type = response.headers.get('Content-Type', '')
        if content_type in ('application/zip', 'application/x-zip-compressed'):
            is_zip = True
        elif image_url.split('/')[-1].split('?')[0].lower().endswith('.zip'):
            is_zip = True
        elif zipfile.is_zipfile(temp_file):
            is_zip = True

        if is_zip:
            # 处理ZIP压缩包
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
            image_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_extensions:
                        image_files.append(os.path.join(root, file))

            if not image_files:
                print("⚠️ 压缩包中未找到任何图片文件")
                return []

            # 确定基础名称（不含扩展名）
            if save_filename:
                base_name = os.path.splitext(save_filename)[0]  # 去掉扩展名
            else:
                # 从URL提取基础名（去掉扩展名）
                url_basename = os.path.basename(urlparse(image_url).path).split('?')[0]
                if url_basename and '.' in url_basename:
                    base_name = os.path.splitext(url_basename)[0]
                else:
                    base_name = "image"

            # 立绘约定 save_filename 已是 {expression}_1.png；ZIP 内多图会再拼 _{idx}，
            # 若直接用 stem「anger_1」会得到 anger_1_1.png。去掉 stem 末尾 _数字 再编号：
            # anger_1 → anger → anger_1.png、anger_2.png…
            m = re.match(r"^(.+)_(\d+)$", base_name)
            zip_series_stem = m.group(1) if m else base_name

            # 排序后重命名
            image_files.sort(key=lambda p: os.path.basename(p))
            saved_paths = []
            for idx, src_path in enumerate(image_files, start=1):
                ext = os.path.splitext(src_path)[1].lower()
                # 扩展名不在支持列表时默认.jpg
                if ext not in image_extensions:
                    ext = '.jpg'
                new_filename = f"{zip_series_stem}_{idx}{ext}"
                dest_path = os.path.join(save_dir, new_filename)
                shutil.move(src_path, dest_path)
                saved_paths.append(dest_path)
                print(f"✅ 图片 {idx} 保存成功：{dest_path}")

            print(f"✅ 共解压并保存 {len(saved_paths)} 张图片")
            return saved_paths

        else:
            # 处理单张图片
            if not save_filename:
                url_filename = os.path.basename(urlparse(image_url).path)
                if url_filename and '.' in url_filename:
                    save_filename = url_filename.split('?')[0]
                else:
                    # 根据Content-Type推断扩展名
                    ext_map = {
                        'image/jpeg': '.jpg',
                        'image/png': '.png',
                        'image/gif': '.gif',
                        'image/webp': '.webp'
                    }
                    ext = ext_map.get(content_type.split(';')[0], '.png')
                    save_filename = f"image_{int(time.time())}{ext}"

            save_path = os.path.join(save_dir, save_filename)
            shutil.move(temp_file, save_path)
            print(f"✅ 图片下载成功：{save_path}")
            return [save_path]

    except requests.exceptions.RequestException as e:
        print(f"❌ 网络请求失败：{str(e)}")
    except zipfile.BadZipFile:
        print(f"❌ 文件损坏或不是有效的ZIP压缩包")
    except Exception as e:
        print(f"❌ 未知错误：{str(e)}")
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except OSError:
                pass
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError:
                pass

    return []


def query_runninghub_result(task_id: str, save_dir: str, save_filename: str):
    conn = http.client.HTTPSConnection("www.runninghub.cn")
    payload = json.dumps({
        "taskId": str(task_id)
    })
    headers = {
        'Authorization': const.running_hub_key,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/openapi/v2/query", payload, headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    json_data = json.loads(data)
    print(json_data)

    # 核心修复点1：先判断status，再处理results列表
    if json_data.get("status") in ("success", "SUCCESS"):
        # 核心修复点2：results是列表，取第一个元素再获取url
        results_list = json_data.get("results", [])
        if results_list:  # 先校验列表非空，避免索引越界
            url = results_list[0].get("url")  # 取列表第一个元素的url
            if url:  # 校验url非空
                download_image_from_url(
                    image_url=url,
                    save_dir=save_dir,
                    save_filename=save_filename,
                )
                return True
            else:
                print("❌ 任务结果中未找到有效URL")
        else:
            print("❌ 任务结果列表为空")
    else:
        print("waiting for task to finish")
    return False  # 明确返回False，避免隐式返回None


def download_runninghub(
    workflowId: str,
    positive_prompt: list[str],
    save_dir: str,
    expressions: Optional[list[str]] = None,
):
    num = len(positive_prompt)
    if expressions is None:
        expressions = [str(i) for i in range(num)]
    if len(expressions) != num:
        raise ValueError("expressions 与 positive_prompt 长度必须一致")

    for i in range(num):
        task_id = run_RunningHub(workflowId=workflowId, positive_prompt=positive_prompt[i])
        if not task_id:
            print(f"❌ 第 {i + 1} 条任务创建失败，已跳过")
            continue

        save_filename = stand_pic_save_filename(expressions[i])

        time.sleep(30)
        while True:
            if query_runninghub_result(
                task_id=task_id,
                save_dir=save_dir,
                save_filename=save_filename,
            ):
                break
            time.sleep(10)


def get_art_prompt(description: str):
    class ArtPrompt(BaseModel):
        art_prompt: str = Field(description="对给的自然语言描述的标准分词描述")

    parser = PydanticOutputParser(pydantic_object=ArtPrompt)
    prompt = PromptTemplate(
        template="""
            你是一个 ComfyUI 提示词专家。你的任务是将用户对人物图片的自然语言描述，转换成适合 Stable Diffusion / ComfyUI 使用的英文提示词。

    要求：
    1. 只输出英文提示词，用英文逗号分隔，不要加解释。
    2. 按此顺序组织：画面质量词 → 主体（人物特征、服饰、动作）→ 背景/环境 → 光照 → 风格/氛围。
    3. 使用 SD 社区常见的标签风格，如：masterpiece, best quality, 1girl, solo, 具体特征使用下划线连接（如 blue_eyes, long_hair）。
    4. 如果描述中包含负面内容（如模糊、畸形等），将其单独整理为负面提示词，用“负面提示词：”标注。
    5. 保持简洁，不输出自然语言句子。
    6. 只用给出相应的active prompt
    7. 仅描述人物特征， 背景描述为简短背景或纯色背景
    示例：
    输入：一个年轻女孩，长发，穿着白色连衣裙，站在樱花树下，阳光透过花瓣洒下来，画风是二次元。
    输出：
    masterpiece, best quality, 1girl, solo, long_hair, white_dress, standing, simple_background, outdoors, sunlight, dappled_light, anime_style


            创作要求：{text}

            请严格按照以下格式输出,不要添加任何额外说明：：
            {format_instructions}

            输出：""",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    llm = llm_chat.LlmChat(
        model_name=const.llm_model,
        temperature=0,
        model_provider='openai',
        base_url=const.llm_base_url,
        api_key=const.api_key or const.ds_api_key or '',
        pydantic_object=ArtPrompt,
        prompt_template=prompt
    )
    llm.structured_chat(description)
    print(llm.new_message)
    return llm.new_message.art_prompt



def default_runninghub_pic_dir() -> str:
    """项目 public/sources/pic，与前端静态资源 /sources/pic 一致。"""
    return str(tools.get_project_root() / "public" / "sources" / "pic")


def get_running_pic(
    workflowId: str,
    positive_prompt: str,
    character_name: str = "",
    save_dir: Optional[str] = None,
):
    """
    图片保存到 public/sources/pic/{角色名}/，文件名为 {expression}_1.png（expression 来自 EXPRESSION_LS）。
    """
    base_dir = save_dir if save_dir else default_runninghub_pic_dir()
    safe_char = sanitize_character_name_for_path(character_name)
    save_dir_final = os.path.join(base_dir, safe_char)
    os.makedirs(save_dir_final, exist_ok=True)

    pm = []
    expression_ls = list(EXPRESSION_LS)
    ex = get_art_prompt(positive_prompt)
    for expression in expression_ls:
        pm.append(f"{ex}, cowboy_shot, {expression}")

    download_runninghub(
        workflowId=workflowId,
        positive_prompt=pm,
        save_dir=save_dir_final,
        expressions=expression_ls,
    )
