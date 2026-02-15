
"""
实战项目：文章写作工作流（LangChain 1.0 + LangGraph）

功能：
- 智能规划大纲
- 分段撰写
- 自动优化
- 质量评估
- 人工审核
"""

import os
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import json
import const

# ==================== 数据模型 ====================

class OutlineSection(BaseModel):
    """大纲章节"""
    title: str = Field(description="章节标题")
    key_points: List[str] = Field(description="关键要点")

class ArticleOutline(BaseModel):
    """文章大纲"""
    title: str = Field(description="文章标题")
    introduction: str = Field(description="引言")
    sections: List[OutlineSection] = Field(description="章节列表")
    conclusion: str = Field(description="结论")

class QualityScore(BaseModel):
    """质量评分"""
    clarity: int = Field(description="清晰度1-10", ge=1, le=10)
    coherence: int = Field(description="连贯性1-10", ge=1, le=10)
    depth: int = Field(description="深度1-10", ge=1, le=10)
    overall: int = Field(description="总分1-10", ge=1, le=10)
    feedback: str = Field(description="改进建议")

# ==================== 状态定义 ====================

class WritingState(TypedDict):
    topic: str                          # 主题
    outline: Optional[ArticleOutline]   # 大纲
    sections_content: List[str]         # 各章节内容
    full_article: str                   # 完整文章
    quality_score: Optional[QualityScore] # 质量评分
    revision_count: int                 # 修订次数
    approved: bool                      # 是否批准
    human_feedback: str                 # 人工反馈

# ==================== 工作流节点 ====================

def create_model():
    """创建模型"""
    return init_chat_model(
        "Qwen/Qwen3-8B",
        model_provider="openai",
        base_url="https://api.siliconflow.cn/v1",
        api_key= const.api_key,
        temperature=0.7
    )

def plan_outline(state: WritingState) -> WritingState:
    """规划大纲"""
    print(f"\n📋 规划大纲: {state['topic']}")

    model = create_model()
    structured_model = model.with_structured_output(ArticleOutline)

    prompt = f"""请为以下主题创建详细的文章大纲：

      主题：{state['topic']}

      要求：
      1. 创建吸引人的标题
      2. 撰写引言（2-3句）
      3. 设计 3-5 个章节，每个章节列出 2-3 个要点
      4. 撰写结论（2-3句）

      请确保逻辑清晰、结构完整。
      """

    outline = structured_model.invoke(prompt)
    state["outline"] = outline

    print(f"✅ 大纲创建完成")
    print(f"   标题: {outline.title}")
    print(f"   章节数: {len(outline.sections)}")

    return state

def write_sections(state: WritingState) -> WritingState:
    """撰写各章节"""
    print(f"\n✍️  撰写文章内容...")

    model = create_model()
    outline = state["outline"]

    sections_content = []

    # 引言
    print("   - 引言")
    sections_content.append(f"## 引言\n\n{outline.introduction}")

    # 各章节
    for i, section in enumerate(outline.sections, 1):
        print(f"   - {section.title}")

        prompt = f"""请撰写文章的这一章节：

        章节标题：{section.title}

        关键要点：
        {chr(10).join([f'- {point}' for point in section.key_points])}

        要求：
        1. 内容详实，每个要点都要充分展开
        2. 语言流畅，逻辑清晰
        3. 字数 300-500 字
        4. 使用 Markdown 格式

        正文：
        """

        content = model.invoke(prompt).content
        sections_content.append(f"## {section.title}\n\n{content}")

    # 结论
    print("   - 结论")
    sections_content.append(f"## 结论\n\n{outline.conclusion}")

    state["sections_content"] = sections_content

    print(f"✅ 内容撰写完成（共 {len(sections_content)} 部分）")

    return state

def assemble_article(state: WritingState) -> WritingState:
    """组装完整文章"""
    print(f"\n🔧 组装文章...")

    outline = state["outline"]
    sections = state["sections_content"]

    # 组装文章
    article = f"# {outline.title}\n\n"
    article += "\n\n".join(sections)
    article += "\n\n---\n\n"
    article += f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"

    state["full_article"] = article

    print(f"✅ 文章组装完成（共 {len(article)} 字符）")

    return state

def evaluate_quality(state: WritingState) -> WritingState:
    """评估文章质量"""
    print(f"\n📊 评估文章质量...")

    model = create_model()
    structured_model = model.with_structured_output(QualityScore)

    prompt = f"""请评估以下文章的质量：

      {state['full_article']}

      评估维度：
      1. 清晰度（1-10）：语言是否清晰易懂
      2. 连贯性（1-10）：逻辑是否连贯
      3. 深度（1-10）：内容是否有深度
      4. 总分（1-10）：综合评分

      请提供具体的改进建议。
      """

    score = structured_model.invoke(prompt)
    state["quality_score"] = score

    print(f"✅ 质量评估完成")
    print(f"   总分: {score.overall}/10")
    print(f"   清晰度: {score.clarity}/10")
    print(f"   连贯性: {score.coherence}/10")
    print(f"   深度: {score.depth}/10")

    return state

def human_review_node(state: WritingState) -> WritingState:
    """人工审核"""
    print(f"\n" + "="*70)
    print("👤 人工审核")
    print("="*70)

    print(f"\n文章标题: {state['outline'].title}")
    print(f"质量评分: {state['quality_score'].overall}/10")
    print(f"\n预览前 500 字符:\n")
    print(state['full_article'][:500])
    print("\n...")

    print(f"\n自动评估建议:")
    print(state['quality_score'].feedback)

    print(f"\n请审核:")
    print("1. 批准发布（输入 'y'）")
    print("2. 需要修订（输入 'n'）")
    print("3. 自动批准所有后续（输入 'auto'）")

    choice = input("\n您的决定: ").strip().lower()

    if choice == 'y' or choice == 'auto':
        state["approved"] = True
        state["human_feedback"] = ""
        print("✅ 已批准")
    else:
        state["approved"] = False
        state["human_feedback"] = input("\n请提供修改意见: ")
        print(f"📝 反馈已记录")

    return state

def revise_article(state: WritingState) -> WritingState:
    """修订文章"""
    print(f"\n🔧 根据反馈修订文章...")
    print(f"   反馈: {state['human_feedback']}")

    model = create_model()

    prompt = f"""请根据以下反馈修订文章：

      原文：
      {state['full_article']}

      修改意见：
      {state['human_feedback']}

      要求：
      1. 针对性改进
      2. 保持原有结构
      3. 提升整体质量

      修订后的文章：
      """

    revised = model.invoke(prompt).content
    state["full_article"] = revised
    state["revision_count"] += 1

    print(f"✅ 修订完成（第 {state['revision_count']} 次）")

    return state

def save_article(state: WritingState) -> WritingState:
    """保存文章"""
    print(f"\n💾 保存文章...")

    filename = f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(state['full_article'])

    # 保存元数据
    metadata = {
        "topic": state["topic"],
        "title": state["outline"].title,
        "quality_score": state["quality_score"].dict(),
        "revision_count": state["revision_count"],
        "generated_at": datetime.now().isoformat()
    }

    meta_filename = filename.replace('.md', '_meta.json')
    with open(meta_filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ 文章已保存")
    print(f"   文件: {filename}")
    print(f"   元数据: {meta_filename}")

    return state

# ==================== 路由函数 ====================

def check_quality(state: WritingState) -> str:
    """检查质量是否达标"""
    score = state["quality_score"].overall

    if score >= 8:
        print("🌟 质量优秀，进入审核")
        return "review"
    elif state["revision_count"] >= 2:
        print("⚠️  已达最大修订次数，进入审核")
        return "review"
    else:
        print("📝 质量需改进，自动修订")
        # 使用质量评估的反馈作为修订意见
        state["human_feedback"] = state["quality_score"].feedback
        return "revise"

def check_approval(state: WritingState) -> str:
    """检查是否批准"""
    return "save" if state["approved"] else "revise"

# ==================== 构建工作流 ====================

def create_writing_workflow():
    """创建写作工作流"""

    workflow = StateGraph(WritingState)

    # 添加节点
    workflow.add_node("plan", plan_outline)
    workflow.add_node("write", write_sections)
    workflow.add_node("assemble", assemble_article)
    workflow.add_node("evaluate", evaluate_quality)
    workflow.add_node("review", human_review_node)
    workflow.add_node("revise", revise_article)
    workflow.add_node("save", save_article)

    # 设置入口
    workflow.set_entry_point("plan")

    # 添加边
    workflow.add_edge("plan", "write")
    workflow.add_edge("write", "assemble")
    workflow.add_edge("assemble", "evaluate")

    # 质量检查的条件分支
    workflow.add_conditional_edges(
        "evaluate",
        check_quality,
        {
            "review": "review",
            "revise": "revise"
        }
    )

    # 人工审核的条件分支
    workflow.add_conditional_edges(
        "review",
        check_approval,
        {
            "save": "save",
            "revise": "revise"
        }
    )

    # 修订后重新评估
    workflow.add_edge("revise", "assemble")

    # 保存后结束
    workflow.add_edge("save", END)

    return workflow.compile()

# ==================== 主程序 ====================

def main():
    """主程序"""
    print("="*70)
    print("📝 智能文章写作工作流（LangChain 1.0 + LangGraph）")
    print("="*70)

    # 创建工作流
    app = create_writing_workflow()
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))



    # 输入主题
    topics = [
        "人工智能在教育领域的应用与挑战",
        "如何使用 LangChain 构建 AI 应用"
    ]

    for topic in topics:
        print(f"\n\n" + "="*70)
        print(f"🎯 开始写作: {topic}")
        print("="*70)

        # 初始化状态
        initial_state = {
            "topic": topic,
            "outline": None,
            "sections_content": [],
            "full_article": "",
            "quality_score": None,
            "revision_count": 0,
            "approved": False,
            "human_feedback": ""
        }

        # 执行工作流
        result = app.invoke(initial_state)

        print(f"\n\n" + "="*70)
        print("✅ 写作完成！")
        print("="*70)
        print(f"主题: {result['topic']}")
        print(f"标题: {result['outline'].title}")
        print(f"质量评分: {result['quality_score'].overall}/10")
        print(f"修订次数: {result['revision_count']}")
        print(f"状态: {'已发布' if result['approved'] else '待处理'}")

        input("\n按 Enter 继续下一篇...")
    # 使用 Graphviz 渲染（Colab 最稳定的方案）
    try:
        display(Image(app.get_graph(xray=True).draw_png()))
    except Exception as e:
        print(f"Graphviz 渲染失败: {e}")
        print("\n使用 Mermaid 文本方式显示:")
        print(app.get_graph(xray=True).draw_mermaid())

if __name__ == "__main__":
    main()