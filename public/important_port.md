# 重要接口文档

## 基础信息

- **API 基础 URL**: `http://localhost:8000`
- **内容类型**: `application/json`
- **前端调用位置**: `src/components/StoryPage.vue`

---

## 1. 一体化故事生成接口（推荐）

### 接口信息
- **路径**: `/api/story/generate`
- **方法**: `POST`
- **标签**: `story`
- **后端路由文件**: `server/app/routers/story.py`
- **特性**: 支持 Server-Sent Events (SSE) 流式输出

### 请求参数

```json
{
  "user_input": "故事大纲内容",
  "strict_model": false
}
```

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| user_input | string | 是 | - | 故事大纲内容（可包含角色信息） |
| strict_model | boolean | 否 | false | 是否启用严格模式 |

### 响应格式（SSE 流式）

接口返回 Server-Sent Events 流，每个事件格式如下：

```
data: {"stage": "start", "message": "开始生成故事...", "progress": 0}

data: {"stage": "outline", "message": "正在生成大纲...", "progress": 10}

data: {"stage": "outline_complete", "message": "大纲生成完成", "progress": 30, "data": {"name": "大纲标题", "content": "大纲内容"}}

data: {"stage": "script", "message": "正在生成剧本...", "progress": 40}

data: {"stage": "script_complete", "message": "剧本生成完成", "progress": 60, "data": {"name": "剧本标题", "site": "地点", "content": "剧本内容"}}

data: {"stage": "dialogue", "message": "正在生成对话剧本...", "progress": 70}

data: {"stage": "dialogue_complete", "message": "对话剧本生成完成", "progress": 90, "data": {"content": "对话剧本内容"}, "dialogue_results": [ ... ]}

data: {"stage": "complete", "message": "故事生成完成！", "progress": 100, "final_result": "最终对话剧本内容", "dialogue_results": [ ... ]}

data: {"stage": "error", "message": "错误信息", "progress": 0}
```

### 生成阶段说明

| 阶段 | 进度 | 说明 |
|------|------|------|
| start | 0% | 开始生成 |
| outline | 10% | 正在生成大纲 |
| outline_complete | 30% | 大纲生成完成 |
| script | 40% | 正在生成剧本 |
| script_complete | 60% | 剧本生成完成 |
| dialogue | 70% | 正在生成对话剧本 |
| dialogue_complete | 90% | 对话剧本生成完成 |
| complete | 100% | 全部完成 |
| error | 0% | 发生错误 |

**`dialogue_results`（对话完成阶段起附带）**：与 `/api/dialogue/generate` 返回的 `results` 同结构的数组，每项含 `chapter_name`、`site`、`dialogues`（`dialogues` 内为 `name`、`dialogue_content`、`character` 等）。故事页「导出并打开流程图」依赖该字段；也可自行调用下文「导出结构化流程 JSON」接口。

### 前端调用示例

```javascript
async function generateStory() {
  const response = await fetch('http://localhost:8000/api/story/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      user_input: storyContent.value,
      strict_model: strictModel.value
    })
  })
  
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    
    const chunk = decoder.decode(value)
    const lines = chunk.split('\n')
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.substring(6))
        
        // 更新进度
        if (data.stage) {
          currentStage.value = data.message
          progress.value = data.progress
        }
        
        // 获取最终结果
        if (data.stage === 'complete' && data.final_result) {
          storyContent.value = data.final_result
        }
        
        // 错误处理
        if (data.stage === 'error') {
          throw new Error(data.message)
        }
      }
    }
  }
}
```

### 优势

1. **一键生成**: 用户只需输入大纲，系统自动完成三步生成
2. **实时反馈**: 通过 SSE 流式输出，实时显示生成进度
3. **简化界面**: 前端只需一个文本框，降低使用复杂度
4. **自动保存**: 每个阶段的结果自动保存到对应目录

---

## 1.1 导出结构化流程 JSON（流程图数据源）

将一键生成得到的 `dialogue_results` 转为流程画布使用的 JSON，并写入 `public/sources/strctured_json/`。实现见 `server/app/services/get_strctured_json.py` 的 `structured_json`，由本接口调用。

### 接口信息

- **路径**: `/api/story/export-structured`
- **方法**: `POST`
- **标签**: `story`
- **后端路由文件**: `server/app/routers/story.py`
- **前端调用位置**: `src/components/StoryPage.vue`（按钮「导出并打开流程图」）

### 请求参数

```json
{
  "dialogue_results": [
    {
      "chapter_name": "章节名",
      "site": "地点描述",
      "dialogues": [
        {
          "name": "角色或旁白",
          "dialogue_content": "台词",
          "character": "微笑"
        }
      ]
    }
  ]
}
```

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| dialogue_results | array | 是 | 与对话生成结果一致；通常来自 SSE 的 `dialogue_results` |

### 响应格式

```json
{
  "public_url": "/sources/strctured_json/renai_20260324_153000.json",
  "file_name": "renai_20260324_153000.json",
  "save_path": "C:\\...\\public\\sources\\strctured_json\\renai_20260324_153000.json"
}
```

| 字段名 | 类型 | 说明 |
|--------|------|------|
| public_url | string | 前端静态资源路径，可用 `fetch(public_url)` 加载 |
| file_name | string | 文件名 |
| save_path | string | 服务端磁盘绝对路径（调试用） |

### 与流程图联动

1. 故事页请求本接口成功后，使用返回的 `public_url` 调用 `FlowCanvas` 暴露的 `loadFromPublicUrl(url)`（见 `src/components/FlowCanvas.vue`）。
2. `App.vue` 中通过 `@open-flow` 切换至流程 Tab 并执行加载。

### 前端调用示例

```javascript
const res = await fetch('http://localhost:8000/api/story/export-structured', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ dialogue_results: lastDialogueResults })
})
const { public_url } = await res.json()
// 切换流程页后: flowCanvasRef.loadFromPublicUrl(public_url)
```

---

## 2. 大纲生成接口（独立调用）

### 接口信息
- **路径**: `/api/outline/generate`
- **方法**: `POST`
- **标签**: `outline`
- **后端路由文件**: `server/app/routers/outline.py`

### 请求参数

```json
{
  "user_input": "故事大纲要求描述",
  "strict_model": false
}
```

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| user_input | string | 是 | - | 用户输入的大纲要求 |
| strict_model | boolean | 否 | false | 是否启用严格模式 |

### 响应格式

```json
{
  "article_outline_name": "文章大纲标题",
  "content": "文章大纲内容"
}
```

| 字段名 | 类型 | 说明 |
|--------|------|------|
| article_outline_name | string | 文章大纲的名字 |
| content | string | 文章大纲的内容 |

### 前端调用示例

```javascript
const response = await fetch('http://localhost:8000/api/outline/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    user_input: outlineContent.value,
    strict_model: strictModel.value
  })
})

const result = await response.json()
// result.article_outline_name - 大纲标题
// result.content - 大纲内容
```

---

## 2. 剧本生成接口

### 接口信息
- **路径**: `/api/script/generate`
- **方法**: `POST`
- **标签**: `script`
- **后端路由文件**: `server/app/routers/script.py`

### 请求参数

```json
{
  "user_input": "剧本要求描述",
  "outline": "故事大纲内容",
  "strict_model": false
}
```

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| user_input | string | 是 | - | 用户输入的剧本要求 |
| outline | string | 是 | - | 故事大纲内容 |
| strict_model | boolean | 否 | false | 是否启用严格模式 |

### 响应格式

```json
{
  "article_script_name": "剧本标题",
  "paragraph_num": 5,
  "content": ["段落1内容", "段落2内容", "..."],
  "site": "地点名称"
}
```

| 字段名 | 类型 | 说明 |
|--------|------|------|
| article_script_name | string | 剧本的名字 |
| paragraph_num | integer | 剧本的分段数目 |
| content | array[string] | 剧本的完整内容，按段落分 |
| site | string | 该分段的地点名 |

### 前端调用示例

```javascript
const response = await fetch('http://localhost:8000/api/script/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    user_input: scriptContent.value,
    outline: outlineContent.value,
    strict_model: scriptStrictModel.value
  })
})

const result = await response.json()
// result.article_script_name - 剧本标题
// result.paragraph_num - 段落数量
// result.content - 内容数组
// result.site - 地点名称
```

---

## 3. 对话剧本生成接口

### 接口信息
- **路径**: `/api/dialogue/generate`
- **方法**: `POST`
- **标签**: `dialogue`
- **后端路由文件**: `server/app/routers/dialogue.py`

### 请求参数

```json
{
  "user_input": "对话剧本要求描述",
  "script_content": "剧本内容",
  "strict_model": false
}
```

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| user_input | string | 是 | - | 用户输入的对话剧本要求 |
| script_content | string | 是 | - | 剧本内容 |
| strict_model | boolean | 否 | false | 是否启用严格模式 |

### 响应格式

```json
{
  "results": [
    {
      "chapter_name": "章节名称",
      "site": "地点描述",
      "dialogues": [
        {
          "name": "角色名称",
          "dialogue_content": "对话内容"
        }
      ]
    }
  ]
}
```

| 字段名 | 类型 | 说明 |
|--------|------|------|
| results | array | 对话结果数组 |
| results[].chapter_name | string | 章节名称 |
| results[].site | string | 地点描述 |
| results[].dialogues | array | 对话列表 |
| results[].dialogues[].name | string | 说话者的名字（旁白则为"旁白"） |
| results[].dialogues[].dialogue_content | string | 对话内容（单句不超过15字） |

### 前端调用示例

```javascript
const response = await fetch('http://localhost:8000/api/dialogue/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    user_input: dialogueContent.value,
    script_content: scriptContent.value,
    strict_model: dialogueStrictModel.value
  })
})

const result = await response.json()
// result.results - 对话结果数组
// result.results[i].chapter_name - 章节名称
// result.results[i].site - 地点
// result.results[i].dialogues - 对话列表
```

---

## 4. 其他接口

### 4.1 标签数据加载接口

- **路径**: `/sources/label2.json`
- **方法**: `GET`
- **调用位置**: `src/components/FlowCanvas.vue:585`
- **用途**: 加载流程画布的标签数据

```javascript
const response = await fetch('/sources/label2.json')
```

### 4.2 编辑器打开接口（开发环境）

- **路径**: `/__open-in-editor?file=README.md`
- **方法**: `GET`
- **调用位置**: `src/components/TheWelcome.vue:9`
- **用途**: 在编辑器中打开指定文件（仅开发环境）

```javascript
const openReadmeInEditor = () => fetch('/__open-in-editor?file=README.md')
```

---

## 5. 人物卡数据接口

### 存储方式

人物卡数据存储在浏览器的 **localStorage** 中，不通过后端 API 获取。

- **存储键名**: `characters_data`
- **存储位置**: 浏览器本地存储
- **调用位置**: 
  - `src/components/CharacterPage.vue` (角色管理页面)
  - `src/components/StoryPage.vue` (故事生成页面)

### 数据结构

```javascript
// localStorage 中存储的完整数据结构
[
  {
    "id": "lxyz123abc456",  // 唯一标识符
    "data": {
      "name": "角色姓名",
      "age": "年龄",
      "gender": "性别",
      "appearance": "外貌描述",
      "personality": "性格特点",
      "background": "背景故事",
      "dialogue_examples": "对话示例",
      "other_settings": "其他设置",
      "metadata": {}  // 元数据
    },
    "images": [
      {
        "id": "img_123",
        "url": "data:image/png;base64,...",  // Base64 编码的图片
        "name": "character_card.png"
      }
    ]
  }
]
```

### 获取人物卡列表

```javascript
// 从 localStorage 获取所有人物卡
function loadCharactersFromStorage() {
  try {
    const saved = localStorage.getItem('characters_data')
    if (saved) {
      const characters = JSON.parse(saved)
      return characters
    }
    return []
  } catch (error) {
    console.error('加载角色数据失败:', error)
    return []
  }
}

// 使用示例
const characters = loadCharactersFromStorage()
console.log('人物卡列表:', characters)
```

### 保存人物卡数据

```javascript
// 保存人物卡到 localStorage
function saveCharactersToStorage(characters) {
  try {
    localStorage.setItem('characters_data', JSON.stringify(characters))
    return true
  } catch (error) {
    console.error('保存角色数据失败:', error)
    return false
  }
}

// 使用示例
const newCharacter = {
  id: Date.now().toString(36) + Math.random().toString(36).substr(2),
  data: {
    name: '新角色',
    age: '25',
    gender: '女',
    appearance: '长发，蓝色眼睛',
    personality: '温柔善良',
    background: '来自神秘的东方国度',
    dialogue_examples: '你好，很高兴认识你。',
    other_settings: '',
    metadata: {}
  },
  images: []
}

characters.push(newCharacter)
saveCharactersToStorage(characters)
```

### 导入角色卡（PNG格式）

角色卡支持从 PNG 图片文件导入，数据嵌入在 PNG 的元数据块中。

```javascript
// 导入角色卡 PNG 文件
async function importCharacterCard(file) {
  const reader = new FileReader()
  
  reader.onload = (e) => {
    const arrayBuffer = e.target.result
    const uint8Array = new Uint8Array(arrayBuffer)
    
    // 验证 PNG 签名
    const pngSignature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    const isPng = pngSignature.every((byte, index) => uint8Array[index] === byte)
    
    if (!isPng) {
      throw new Error('不是有效的PNG文件')
    }
    
    // 解析 PNG 文本块
    let textChunks = []
    let offset = 8
    
    while (offset < uint8Array.length) {
      const length = (uint8Array[offset] << 24) | (uint8Array[offset + 1] << 16) | 
                     (uint8Array[offset + 2] << 8) | uint8Array[offset + 3]
      const type = String.fromCharCode(...uint8Array.slice(offset + 4, offset + 8))
      
      if (type === 'tEXt' || type === 'iTXt') {
        const chunkData = uint8Array.slice(offset + 8, offset + 8 + length)
        const nullIndex = chunkData.findIndex(byte => byte === 0)
        const keyword = String.fromCharCode(...chunkData.slice(0, nullIndex))
        const text = new TextDecoder('utf-8').decode(chunkData.slice(nullIndex + 1))
        textChunks.push({ keyword, text })
      }
      
      offset += 12 + length
    }
    
    // 查找角色数据块（支持多种格式）
    const charaChunk = textChunks.find(c => c.keyword === 'chara') || 
                       textChunks.find(c => c.keyword === 'ccv3') ||
                       textChunks.find(c => c.keyword === 'ccv2')
    
    if (!charaChunk) {
      throw new Error('未找到角色数据')
    }
    
    // 解码 Base64 数据
    let jsonStr
    try {
      jsonStr = atob(charaChunk.text)
    } catch {
      jsonStr = charaChunk.text
    }
    
    const characterData = JSON.parse(jsonStr)
    
    // 将图片转为 Base64
    let binary = ''
    for (let i = 0; i < uint8Array.length; i++) {
      binary += String.fromCharCode(uint8Array[i])
    }
    const base64Png = btoa(binary)
    
    return {
      data: characterData,
      images: [{
        id: Date.now().toString(36),
        url: 'data:image/png;base64,' + base64Png,
        name: file.name || 'character_card'
      }]
    }
  }
  
  reader.readAsArrayBuffer(file)
}
```

### 导出角色卡（PNG格式）

```javascript
// 导出角色卡为 PNG 文件
function exportCharacterCard(character) {
  // 下载图片
  if (character.images && character.images[0]) {
    const link = document.createElement('a')
    link.href = character.images[0].url
    link.download = `${character.data.name || 'character'}_card.png`
    link.click()
  }
}
```

### 在故事生成中使用人物卡

在 `StoryPage.vue` 中，可以通过 `@` 符号快速引用人物卡：

```javascript
// 打开角色选择器
function openCharacterSelector(editor) {
  activeEditor.value = editor
  showCharacterSelector.value = true
}

// 插入角色引用
function insertCharacterReference(character) {
  const ref = `@${character.data.name || '角色'}`
  const editor = activeEditor.value
  
  if (editor === 'outline') {
    // 在大纲编辑器中插入
    const textarea = document.querySelector('.outline-editor')
    const start = textarea.selectionStart
    const end = textarea.selectionEnd
    const text = outlineContent.value
    outlineContent.value = text.substring(0, start) + ref + text.substring(end)
  }
  // ... 其他编辑器类似处理
  
  closeCharacterSelector()
}
```

### 人物卡字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | string | 唯一标识符 |
| data.name | string | 角色姓名 |
| data.age | string | 年龄 |
| data.gender | string | 性别 |
| data.appearance | string | 外貌描述 |
| data.personality | string | 性格特点 |
| data.background | string | 背景故事 |
| data.dialogue_examples | string | 对话示例 |
| data.other_settings | string | 其他设置 |
| data.metadata | object | 元数据对象 |
| images | array | 图片数组 |
| images[].id | string | 图片ID |
| images[].url | string | Base64 编码的图片 URL |
| images[].name | string | 图片名称 |

### 注意事项

1. **存储限制**: localStorage 通常有 5-10MB 的存储限制，大量角色卡可能超出限制
2. **数据持久性**: localStorage 数据仅存储在当前浏览器，清除浏览器数据会丢失
3. **图片大小**: Base64 编码会使图片体积增大约 33%，建议优化图片大小
4. **跨域限制**: localStorage 遵循同源策略，不同域名无法共享数据
5. **备份建议**: 建议定期导出角色卡进行备份

---

## 错误处理

所有接口在发生错误时返回 HTTP 500 状态码，响应格式：

```json
{
  "detail": "错误信息描述"
}
```

前端统一错误处理示例：

```javascript
try {
  const response = await fetch(url, options)
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }
  
  const result = await response.json()
  // 处理成功响应
} catch (error) {
  console.error('请求失败:', error)
  alert('请求失败: ' + error.message)
}
```

---

## 生成流程说明

### 完整的故事生成流程

1. **第一步：生成大纲**
   - 接口：`/api/outline/generate`
   - 输入：用户的大纲要求
   - 输出：故事大纲

2. **第二步：生成剧本**
   - 接口：`/api/script/generate`
   - 输入：剧本要求 + 第一步生成的大纲
   - 输出：完整剧本（分段）

3. **第三步：生成对话剧本**
   - 接口：`/api/dialogue/generate`
   - 输入：对话要求 + 第二步生成的剧本
   - 输出：对话剧本（包含角色对话）

### 依赖关系

```
大纲生成 → 剧本生成 → 对话剧本生成
```

每个步骤都依赖前一步的输出结果。

---

## 文件保存路径

生成的文件会自动保存到以下目录：

| 接口类型 | 保存目录 | 文件名格式 |
|---------|---------|-----------|
| 大纲 | `public/sources/outline/` | `outline_YYYYMMDD_HHMMSS.json` |
| 剧本 | `public/sources/complete_script/` | `script_YYYYMMDD_HHMMSS.json` |
| 对话剧本 | `public/sources/dialogue/` | `dialogue_YYYYMMDD_HHMMSS.json` |
| 流程图结构化数据 | `public/sources/strctured_json/` | `renai_YYYYMMDD_HHMMSS.json`（由 `/api/story/export-structured` 或 `structured_json` 写入） |

---

## 严格模式说明

`strict_model` 参数控制生成质量要求：

- **false（默认）**: 接受 "Good" 级别的生成结果
- **true**: 只接受 "Perfect" 级别的生成结果，质量要求更高

---

## 开发建议

1. **接口测试**: 使用 Postman 或类似工具测试接口时，确保后端服务已启动在 `http://localhost:8000`
2. **错误日志**: 后端错误会打印到控制台，前端错误会通过 `console.error` 输出
3. **超时处理**: LLM 生成可能耗时较长，建议添加请求超时处理
4. **并发控制**: 避免同时发起多个生成请求，建议使用 `isGenerating` 状态控制
