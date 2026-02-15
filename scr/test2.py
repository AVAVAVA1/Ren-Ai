from typing import List, Optional
from pydantic import BaseModel, Field
import asyncio
from langchain_core.tools import tool

# 带复杂参数的工具
class SearchParams(BaseModel):
    """搜索参数"""
    keyword: str = Field(description="搜索关键词")
    category: Optional[str] = Field(default=None, description="分类筛选")
    max_results: int = Field(default=5, description="最大结果数")

@tool
def advanced_search(params: SearchParams) -> str:
    """高级搜索功能

    支持关键词搜索、分类筛选、结果数量限制

    Args:
        params: 搜索参数对象
    """
    results = f"搜索 '{params.keyword}'"
    if params.category:
        results += f" 在分类 '{params.category}'"
    results += f"，返回前 {params.max_results} 条结果"

    return results


# 返回结构化数据的工具
import json
@tool
def get_product_info(product_id: str) -> str:
    """获取产品详细信息

    Args:
        product_id: 产品ID

    Returns:
        JSON 格式的产品信息
    """
    # 模拟产品数据
    products = {
        "P001": {
            "name": "iPhone 15",
            "price": 5999,
            "stock": 100,
            "rating": 4.8
        },
        "P002": {
            "name": "MacBook Pro",
            "price": 12999,
            "stock": 50,
            "rating": 4.9
        }
    }

    product = products.get(product_id, {"error": "产品不存在"})
    return json.dumps(product, ensure_ascii=False)

# 异步工具
# 对于 I/O 密集型操作（网络请求、数据库查询），使用异步工具：


@tool
async def fetch_data_async(url: str) -> str:
    """异步获取网络数据

    Args:
        url: 目标 URL
    """
    # 模拟网络请求
    await asyncio.sleep(1)
    return f"从 {url} 获取的数据"

# 使用异步 agent
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "获取数据"}]
})