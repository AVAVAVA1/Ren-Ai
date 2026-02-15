import json
from typing import Dict, List, Any  # 类型注解，让代码更规范
import os
def read_json_file(file_path: str, encoding: str = "utf-8") -> Dict | List | Any:
    """
    读取JSON文件，解析为Python的字典/列表对象
    :param file_path: JSON文件的路径（相对路径/绝对路径）
    :param encoding: 文件编码，默认utf-8，需兼容gbk/gb2312可手动指定
    :return: 解析后的Python对象（字典/列表，取决于JSON文件的顶层结构）
    :raises: 捕获并抛出明确的异常信息，方便问题排查
    """
    try:
        # with语句自动管理文件句柄，无需手动close，避免资源泄漏
        with open(file_path, mode="r", encoding=encoding) as f:
            # json.load 直接解析文件对象，区别于json.loads（解析字符串）
            json_data = json.load(f)
        return json_data
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：未找到指定的JSON文件，路径为 {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"错误：JSON文件格式非法，解析失败，详情：{str(e)}")
    except PermissionError:
        raise PermissionError(f"错误：无权限读取文件 {file_path}，请检查文件权限")
    except Exception as e:
        raise Exception(f"读取JSON文件时发生未知错误：{str(e)}")


def save_dict_to_json(
        data: Dict | List[Dict],
        file_path: str,
        encoding: str = "utf-8",
        indent: int = 4,
        ensure_ascii: bool = False
) -> None:
    """
    将单个字典/字典构成的列表保存为JSON文件
    :param data: 待保存数据，支持单个字典、字典组成的列表（数组）
    :param file_path: 保存的JSON文件路径（相对/绝对路径，支持新建文件）
    :param encoding: 文件编码，默认utf-8，兼容gbk/gb2312
    :param indent: JSON格式化缩进空格数，默认4（易读），设为None则为紧凑格式（无缩进）
    :param ensure_ascii: 是否转义中文/特殊字符，默认False（中文原样保存，不转Unicode）
    :return: 无返回值，保存失败则抛出明确异常
    :raises: 捕获并抛出文件/数据相关异常，方便问题排查
    """
    try:
        # 自动创建文件所在的文件夹（如果文件夹不存在，避免路径错误）
        file_dir = os.path.dirname(file_path)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        # with语句自动管理文件句柄，覆盖式写入（JSON常规保存方式）
        with open(file_path, mode="w", encoding=encoding) as f:
            # json.dump 直接将Python对象写入文件对象
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        print(f"数据已成功保存为JSON文件，路径：{file_path}")
    except PermissionError:
        raise PermissionError(f"错误：无权限写入文件 {file_path}，请检查文件/文件夹权限")
    except TypeError as e:
        raise TypeError(f"错误：数据包含JSON不可序列化的对象，详情：{str(e)}（仅支持字典/列表/字符串/数字等基础类型）")
    except OSError as e:
        raise OSError(f"错误：文件路径无效/写入失败，详情：{str(e)}（请检查路径是否包含特殊字符）")
    except Exception as e:
        raise Exception(f"保存JSON文件时发生未知错误：{str(e)}")