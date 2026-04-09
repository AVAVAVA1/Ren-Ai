import http.client
import json
import os
import time
import zipfile
import tempfile
import shutil
import requests
from typing import List, Optional
from urllib.parse import urlparse

from app.services import const

# 背景图提示词统一追加，避免生成带人物的场景图
_BG_NO_PEOPLE = "不要出现人物"


def _append_no_people_to_prompts(prompts: list[str]) -> list[str]:
    """每条 positive_prompt 末尾追加「不要出现人物」（已包含则不再重复）。"""
    out: list[str] = []
    for p in prompts:
        s = (p or "").strip()
        if _BG_NO_PEOPLE in s:
            out.append(s)
        elif s:
            out.append(f"{s}，{_BG_NO_PEOPLE}")
        else:
            out.append(_BG_NO_PEOPLE)
    return out


# 2037179226444533762
def run_RunningHub(workflowId: str, positive_prompt: str):
    api_key = const.running_hub_key
    conn = http.client.HTTPSConnection("www.runninghub.cn")
    payload = json.dumps({
        "apiKey": api_key,
        "workflowId": workflowId,
        "nodeInfoList": [
            {
                "nodeId": "11",
                "fieldName": "编辑文本",
                "fieldValue": positive_prompt
            },
            {
                "nodeId": "12",
                "fieldName": "aspect_ratio",
                "fieldValue": "16:9(1664x928)"
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
    if result_json.get("msg") == "success":
        task_id = result_json['data']['taskId']
        return task_id
    print('task error', result_json)
    return None


def download_image_from_url(
        image_url: str,
        save_dir: str = "../pic/bg",
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

            # 排序后重命名
            image_files.sort(key=lambda p: os.path.basename(p))
            saved_paths = []
            for idx, src_path in enumerate(image_files, start=1):
                ext = os.path.splitext(src_path)[1].lower()
                # 扩展名不在支持列表时默认.jpg
                if ext not in image_extensions:
                    ext = '.jpg'
                new_filename = f"{base_name}_{idx}{ext}"
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
            temp_file = None
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

    if json_data.get("status") in ("success", "SUCCESS"):
        results_list = json_data.get("results", [])
        if results_list:
            url = results_list[0].get("url")
            if url:
                download_image_from_url(image_url=url, save_dir=save_dir, save_filename=save_filename)
                return True
            print("❌ 任务结果中未找到有效URL")
        else:
            print("❌ 任务结果列表为空")
    else:
        print("waiting for task to finish")
    return False


def download_runninghub(
    workflowId: str,
    positive_prompt: list[str],
    save_dir: str,
    save_filenames: list[str],
):
    if len(positive_prompt) != len(save_filenames):
        raise ValueError("positive_prompt 与 save_filenames 长度须一致")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(positive_prompt)):
        task_id = run_RunningHub(workflowId=workflowId, positive_prompt=positive_prompt[i])
        if not task_id:
            raise RuntimeError(f"第 {i + 1} 条 RunningHub 任务创建失败")
        time.sleep(30)
        while True:
            if query_runninghub_result(
                task_id=task_id,
                save_dir=save_dir,
                save_filename=save_filenames[i],
            ):
                break
            time.sleep(10)


def get_bg(
    workflowId: str,
    positive_prompt: list[str],
    save_dir: str,
    save_filenames: list[str],
):
    """按块调用 RunningHub 出背景图；图片写入 save_dir，文件名与 save_filenames 一一对应。"""
    prompts = _append_no_people_to_prompts(positive_prompt)
    download_runninghub(
        workflowId=workflowId,
        positive_prompt=prompts,
        save_dir=save_dir,
        save_filenames=save_filenames,
    )

    #  注意函数
