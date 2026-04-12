# Ren-Ai：LLM、RunningHub 生图与流程 / 人物卡说明

本文档面向在本仓库上开发或部署的同学，**侧重**后端大模型调用链、**RunningHub** 出图管线，并简要说明**流程图**与**人物卡**在前端的角色及与后端的衔接。

---

## 1. 配置总览（`server/.env`）

后端在 `app/services/const.py` 中通过 `python-dotenv` 加载 **`server/.env`**（仅这一处，勿再依赖 `app/services/.env`）。

| 变量 | 作用 |
|------|------|
| `DS_API_KEY` / `LLM_API_KEY` / `OPENAI_API_KEY` | 大模型 API Key（优先 `DS_API_KEY`，兼容 OpenAI 协议） |
| `LLM_API_BASE` / `OPENAI_BASE_URL` | 可选，自定义 API 基地址（默认 DeepSeek） |
| `LLM_MODEL` | 可选，模型名（默认 `deepseek-chat`） |
| `RunningHub_API_KEY` | [RunningHub](https://www.runninghub.cn) 任务 API，用于生成立绘与场景背景 |
| `REMOVE_BG_API_KEY` / `REMOVE_BG_KEY` | remove.bg，人物立绘去背景（可选） |

密钥勿提交到 Git；`.gitignore` 应忽略 `.env`。

---

## 2. LLM 部分

### 2.1 架构要点

- **`LLMGenerator`**（`server/app/services/llm_generator.py`）  
  - 基于 **LangChain** 的 `PromptTemplate` + `PydanticOutputParser`，把模型输出解析成 **Pydantic 模型**。  
  - 内置 **`generate_with_retry`**：解析失败时降温重试，提高大纲/剧本/对话等结构化输出的成功率。  
  - 初始化时若未配置任何 API Key，会抛出明确错误，提示在 `server/.env` 中设置 `DS_API_KEY` 等。

- **`LlmChat`**（`server/app/services/llm_chat.py`）  
  - 实际发起 HTTP 调用（OpenAI 兼容接口），供 `LLMGenerator` 使用。

- **`const`**（`server/app/services/const.py`）  
  - 统一读取 `llm_base_url`、`llm_model`、`api_key` 等，各业务模块无需重复读环境变量。

### 2.2 业务链路（故事生成）

典型流程由 **`/api/story/generate`**（`server/app/routers/story.py`）以 **SSE** 流式推进，内部依次调用例如：

| 阶段 | 服务模块 | 说明 |
|------|-----------|------|
| 大纲 | `get_outline.outline` | 输出文章/作品大纲结构化字段 |
| 完整剧本 | `get_complete_script.complete_script` | 结合大纲生成多段剧本正文 |
| 对话 | `get_dialogue.dialogue` | 生成对话体内容 |
| 结构化 JSON | `get_strctured_json` 等 | 将结果整理为可供流程图 / Ren’Py 使用的结构 |

此外还有独立路由：**大纲** `/api/outline`、**剧本** `/api/script`、**对话** `/api/dialogue` 等，内部同样通过 **`LLMGenerator`** 与上述配置对接。

### 2.3 与 RunningHub 的交叉点（立绘）

**人物立绘**路径里会再用一次 LLM：  
`get_running_pic` → **`get_art_prompt`**（`get_runninghub_pic.py`）把「外貌 + 性格」自然语言转成 **英文 Stable Diffusion 风格提示词**，再拼表情标签送入 RunningHub。**这与故事生成用的是同一套 `const` 里的 LLM 配置。**

---

## 3. RunningHub 生图部分

RunningHub 通过 **HTTPS** 调用 `www.runninghub.cn` 的 OpenAPI：创建任务 → 轮询结果 → 下载图片（或 ZIP 内多图）。**`workflowId` 与 ComfyUI 中节点 id 必须与线上工作流一致**，否则需在代码里改 `nodeInfoList`。

本仓库里存在 **两套** 调用形态（勿混用同一套 node 配置）：

### 3.1 人物立绘（多表情）

- **代码**：`server/app/services/get_runninghub_pic.py`  
- **HTTP 创建任务**：`run_RunningHub` 中示例节点为 **`nodeId` 48（text）**、**27（value）**，与**立绘工作流**绑定。  
- **入口**：`POST /api/image/generate-character-pics`  
  - 请求体：`workflow_id`（与**设置页**保存的 RunningHub 工作流 ID 一致）、`character_name`、`appearance`、`personality`。  
  - 内部：`get_running_pic` → 多表情列表循环出图，保存到 **`public/sources/pic/{角色目录}/`**，如 `happy_1.png`。  
  - 前端静态访问前缀：**`/sources/pic/...`**

- **去背景（可选）**：`POST /api/image/remove-stand-pic-backgrounds`，依赖 remove.bg，覆盖同目录 PNG。

### 3.2 场景背景（流程图区块）

- **代码**：`server/app/services/get_bg.py`  
- **HTTP 创建任务**：`run_RunningHub` 使用 **`nodeId` 11（编辑文本）**、**12（aspect_ratio）**，与**背景工作流**绑定；默认比例示例为 `16:9(1664x928)`。  
- **提示词**：对每条 `site_description` 会在末尾统一追加 **「不要出现人物」**（`_BG_NO_PEOPLE`），减少场景图里出人像。  
- **入口**：`POST /api/image/generate-flow-backgrounds`  
  - 请求体：与流程图「导出 JSON」同结构的 `structured_json`，可选 `workflow_id`（默认与 `get_bg` 注释中的示例 ID 一致）。  
  - 输出目录：**`public/pic_bg/{yyyyMMdd_HHmmss}/地点{i}.png`**，节点 `background` 写入 **`/pic_bg/...`** 供 Vite 静态资源访问。

### 3.3 前端如何选工作流 ID

- **设置页**（`SettingsPage.vue`）将 **RunningHub 工作流 ID** 存 **`localStorage`**（如 `renai_runninghub_workflow_id`），人物卡「生成立绘」请求会带上该 ID。  
- **流程图「一键生成背景」** 当前默认使用代码里配置的 **`workflow_id`**（可与设置页拆成两套工作流：人物 vs 场景）。

---

## 4. 流程图（简要）

- **页面**：`src/components/FlowCanvas.vue`（Vue Flow）。  
- **数据模型**：多「区块」，每块含 `dialogue_name`、`site_description`、`dialogue_content[]`（节点含 `children` / `parent_id` / `checkFlag` / `menu` 等），与 `public/sources/strctured_json` 下 JSON 结构一致。  
- **与后端**：可导入结构化 JSON；**导出 JSON** 与 Navbar「导出」一致；**一键生成背景** 调用上文 **`generate-flow-backgrounds`** 并回写节点 `background`。  
- **预览**：`POST /api/play/run-flow-pygame` 将当前导出结构写入临时文件并启动 **`pygame_play`**（需本机安装 pygame；若 uvicorn 与 pygame 不在同一 Python，可设环境变量 **`RENAI_PYGAME_PYTHON`**）。

---

## 5. 人物卡（简要）

- **页面**：`src/components/CharacterPage.vue`。  
- **数据**：角色卡多存于浏览器 **localStorage**，与流程图侧「流中的角色」校验可配合使用。  
- **生成立绘**：调用 **`/api/image/generate-character-pics`**（`image_backend` 选 RunningHub 或 ComfyUI）；RunningHub 时使用设置中的 **workflow_id**；生成文件落在 **`public/sources/pic/{角色名}/`**，便于与 `pygame_play` 中 `public/sources/pic/{name}/{表情}.png` 规则对齐。

---

## 6. 相关文件速查

| 主题 | 路径 |
|------|------|
| LLM 配置 | `server/app/services/const.py` |
| 通用生成器 | `server/app/services/llm_generator.py`、`llm_chat.py` |
| 大纲 / 剧本 / 对话 | `get_outline.py`、`get_complete_script.py`、`get_dialogue.py` |
| 生图路由（RunningHub / ComfyUI） | `server/app/routers/runninghub.py`（前缀 `/api/image`） |
| 立绘管线 | `server/app/services/get_runninghub_pic.py` |
| 背景管线 | `server/app/services/get_bg.py` |
| Pygame 预览 | `server/app/services/pygame_play.py`、`server/app/routers/play.py` |
| 流程图 UI | `src/components/FlowCanvas.vue` |
| 人物卡 UI | `src/components/CharacterPage.vue` |
| RunningHub 工作流 ID（前端） | `src/components/SettingsPage.vue` |

---

## 7. 调试建议

- LLM 报错优先检查 **`server/.env`** 是否被加载、Key 是否有效、网络是否可达 `LLM_API_BASE`。  
- RunningHub 报错检查 **`RunningHub_API_KEY`**、`workflowId` 是否与控制台工作流一致，以及节点 **11/12**（背景）与 **48/27**（立绘）是否分别匹配。  
- 长时间任务（生图）请增大前端 **fetch 超时** 或改为异步任务设计（当前为同步阻塞式轮询）。

如有接口或字段变更，请以 **`server/app/routers/*.py`** 与对应 **`services`** 实现为准，并同步更新本文档。
